"""
evaluation_pipeline.py
-------------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Aug 13 joint milestone. Chains together everything Person B owns —
    embedding, matching, cross-course allocation, decision, and
    reasoning — into a single function Person A's pipeline can call per
    student evaluation, instead of wiring four separate modules together
    by hand.

    This is what POST /evaluations/{id}/run is expected to call into on
    Person A's side, per the shared interface contract (eligibility ->
    matching -> GPA -> reasoning).

Edge cases explicitly handled here (per the Aug 13 deliverable):
    - Empty / near-empty transcript (student submitted zero or very few
      source topics)
    - Zero topic overlap (student's topics don't match the target course
      at all)
    - Single-topic courses (target course has only one topic to satisfy)
"""

from embeddings.embed_topics import get_embeddings_batch
from matching_engine import match_course, allocate_across_courses
from decision_engine import decide_with_color
from reasoning_generator import generate


def _embed_topic_texts(texts: list[str]) -> list[dict]:
    """Turn raw topic text into the {"text", "embedding"} shape every
    downstream function expects. Returns [] unchanged for an empty list
    rather than calling the model on nothing (handles the empty-
    transcript edge case cleanly at the embedding step itself)."""
    if not texts:
        return []
    embeddings = get_embeddings_batch(texts)
    return [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]


def run_single_course_evaluation(
    target_topic_texts: list[str],
    source_topic_texts: list[str],
    topic_threshold: float = 0.75,
    accept_threshold: float = 80.0,
    partial_threshold: float = 40.0,
    credit_hours_required: float | None = None,
    credit_hours_completed: float | None = None,
) -> dict:
    """
    Full pipeline for ONE target course vs. one student's pooled source
    topics: embed -> match -> decide -> reason.

    Use this when a student is only being evaluated against a single
    target course. For multiple target courses in the same session
    (where cross-course double-counting is possible), use
    run_multi_course_evaluation() instead.

    Args:
        target_topic_texts: raw topic text for the REC course being
            evaluated against (e.g. from the pre-embedded REC syllabus
            in practice — accepted as raw text here for a self-contained
            pipeline call).
        source_topic_texts: raw topic text extracted from the student's
            transcript/syllabus upload. Can be empty (handles the
            empty-transcript edge case).
        topic_threshold / accept_threshold / partial_threshold: tunable
            thresholds, passed straight through to match_course() and
            decide_with_color().
        credit_hours_required / credit_hours_completed: optional, feeds
            reasoning_generator's credit_difference field.

    Returns:
        {
          "match_results": [...],
          "coverage_pct": float,
          "decision": {"status": ..., "color": ...},
          "reasoning": {...}
        }
    """
    # Edge case: target course has zero topics — nothing to evaluate.
    if not target_topic_texts:
        raise ValueError("target_topic_texts is empty — target course has no topics to evaluate.")

    target_topics = _embed_topic_texts(target_topic_texts)
    # Edge case: empty/near-empty transcript. _embed_topic_texts already
    # returns [] for an empty list; match_course() itself handles an
    # empty pooled_topics list by marking everything uncovered (0%)
    # rather than erroring, so this flows through safely end to end.
    pooled_topics = _embed_topic_texts(source_topic_texts)

    match_results, coverage_pct = match_course(target_topics, pooled_topics, topic_threshold)

    decision = decide_with_color(coverage_pct, accept_threshold, partial_threshold)

    reasoning = generate(
        match_results,
        coverage_pct,
        decision["status"],
        credit_hours_required=credit_hours_required,
        credit_hours_completed=credit_hours_completed,
    )

    return {
        "match_results": match_results,
        "coverage_pct": coverage_pct,
        "decision": decision,
        "reasoning": reasoning,
    }


def run_multi_course_evaluation(
    courses: list[dict],
    source_topic_texts: list[str],
    topic_threshold: float = 0.75,
    accept_threshold: float = 80.0,
    partial_threshold: float = 40.0,
) -> dict:
    """
    Full pipeline for MULTIPLE target courses evaluated against the same
    pooled source topics in one session — uses allocate_across_courses()
    so a source topic can't double-count toward more than one course.

    Args:
        courses: [{ "course_id": Any, "target_topic_texts": list[str] }, ...]
        source_topic_texts: student's pooled source topics (shared
            across all courses in this evaluation). Can be empty.

    Returns:
        { course_id: {
            "match_results": [...],
            "coverage_pct": float,
            "decision": {"status": ..., "color": ...},
            "reasoning": {...}
          }, ... }
    """
    if not courses:
        raise ValueError("courses is empty — nothing to evaluate.")

    pooled_topics = _embed_topic_texts(source_topic_texts)

    courses_for_allocation = []
    for course in courses:
        if not course["target_topic_texts"]:
            raise ValueError(f"course_id={course['course_id']} has no target topics.")
        courses_for_allocation.append({
            "course_id": course["course_id"],
            "target_topics": _embed_topic_texts(course["target_topic_texts"]),
        })

    allocation_results = allocate_across_courses(courses_for_allocation, pooled_topics, topic_threshold)

    final_results = {}
    for course_id, (match_results, coverage_pct) in allocation_results.items():
        decision = decide_with_color(coverage_pct, accept_threshold, partial_threshold)
        reasoning = generate(match_results, coverage_pct, decision["status"])

        final_results[course_id] = {
            "match_results": match_results,
            "coverage_pct": coverage_pct,
            "decision": decision,
            "reasoning": reasoning,
        }

    return final_results


# ---------------------------------------------------------------------
# Standalone demo — runs 3 mock transcripts end to end, covering the
# Aug 13 deliverable's required edge cases: full/partial match, an
# empty transcript, and a single-topic target course.
#
# Requires the real sentence-transformers model, so run this on your
# own machine (not testable in this sandbox — no network access to
# download the model here). This is NOT a pytest file; run it directly:
#     python3 evaluation_pipeline.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("MOCK TRANSCRIPT 1 — Partial match, multi-topic target course")
    print("=" * 70)
    result_1 = run_single_course_evaluation(
        target_topic_texts=[
            "Unit 1: Introduction to supervised learning, linear regression",
            "Unit 2: Classification algorithms, logistic regression, decision trees",
            "Unit 5: Neural networks, backpropagation, gradient descent",
        ],
        source_topic_texts=[
            "Introduction to regression analysis and error minimization",
            "Tree-based classifiers and their use in categorization problems",
            # deliberately nothing matching neural networks -> partial coverage
        ],
        credit_hours_required=4,
        credit_hours_completed=3,
    )
    print(json.dumps(result_1, indent=2, default=str))

    print("\n" + "=" * 70)
    print("MOCK TRANSCRIPT 2 — Empty transcript (edge case)")
    print("=" * 70)
    result_2 = run_single_course_evaluation(
        target_topic_texts=["Unit 1: Introduction to supervised learning"],
        source_topic_texts=[],  # student submitted nothing
    )
    print(json.dumps(result_2, indent=2, default=str))
    assert result_2["coverage_pct"] == 0.0
    assert result_2["decision"]["status"] == "redo"
    print("Edge case verified: empty transcript -> 0% coverage, 'redo' status.")

    print("\n" + "=" * 70)
    print("MOCK TRANSCRIPT 3 — Single-topic target course, full match")
    print("=" * 70)
    result_3 = run_single_course_evaluation(
        target_topic_texts=["Unit 1: Introduction to supervised learning, linear regression"],
        source_topic_texts=["Regression basics and predictive modeling fundamentals"],
    )
    print(json.dumps(result_3, indent=2, default=str))

    print("\nAll 3 mock transcripts ran end to end without errors.")
