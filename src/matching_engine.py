"""
matching_engine.py
--------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Compare a target course's topics (REC's requirements, embedded via
    embeddings/embed_rec_syllabus.py) against a student's pooled source
    topics (embedded via embeddings/embed_source_syllabus.py), and
    produce a per-topic match report plus an overall coverage %.

    "Pooling" means the student's topics may come from multiple source
    courses combined into one list before being passed in here — this
    function doesn't care where they came from, only whether they cover
    the target course's content.

This is the exact interface contract Person A's pipeline calls:

    def match_course(target_topics, pooled_topics, topic_threshold=0.75)
        -> tuple[list[dict], float]
"""

import numpy as np


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two embedding vectors, as a float 0-1ish.
    (Embeddings from embed_topics.py are already normalized, so this is
    effectively just a dot product — but computed explicitly here so this
    function works correctly even if that assumption ever changes.)"""
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_course(
    target_topics: list[dict],
    pooled_topics: list[dict],
    topic_threshold: float = 0.75,
) -> tuple[list[dict], float]:
    """
    Match a target course's topics against a student's pooled source topics.

    Args:
        target_topics: [{ "text": str, "embedding": list[float] }, ...]
            — REC's required topics for the course being evaluated.
        pooled_topics: [{ "text": str, "embedding": list[float] }, ...]
            — student's combined submitted topics (possibly from
            multiple source courses).
        topic_threshold: minimum cosine similarity for a target topic to
            count as "covered" by its best match. Default 0.75.

    Returns:
        (per_topic_results, coverage_pct)

        per_topic_results: one entry per target topic —
            [{
                "target_text": str,
                "best_match_text": str | None,
                "similarity": float,
                "covered": bool
            }, ...]

        coverage_pct: percentage of target topics that were "covered"
            (best match similarity >= topic_threshold).
    """
    if not target_topics:
        raise ValueError("target_topics is empty — nothing to evaluate against.")

    per_topic_results = []

    # Edge case: student submitted no topics at all (empty transcript).
    # Every target topic is automatically unmatched, coverage = 0%.
    if not pooled_topics:
        for target in target_topics:
            per_topic_results.append({
                "target_text": target["text"],
                "best_match_text": None,
                "similarity": 0.0,
                "covered": False,
            })
        return per_topic_results, 0.0

    for target in target_topics:
        best_similarity = -1.0
        best_match_text = None

        for source in pooled_topics:
            sim = _cosine_similarity(target["embedding"], source["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_match_text = source["text"]

        covered = best_similarity >= topic_threshold
        per_topic_results.append({
            "target_text": target["text"],
            "best_match_text": best_match_text,
            "similarity": round(best_similarity, 4),
            "covered": covered,
        })

    covered_count = sum(1 for r in per_topic_results if r["covered"])
    coverage_pct = round((covered_count / len(target_topics)) * 100, 2)

    return per_topic_results, coverage_pct


def allocate_across_courses(
    courses: list[dict],
    pooled_topics: list[dict],
    topic_threshold: float = 0.75,
) -> dict:
    """
    Run greedy cross-course allocation so a single source topic can't be
    used to justify credit for more than one target course.

    Without this, calling match_course() separately per target course
    lets the same source topic "double-count": e.g. one broad source
    course could independently satisfy both a Machine Learning target
    AND a Data Mining target, even though the student only studied that
    content once. This function resolves conflicts by giving the
    highest-similarity match first claim on a source topic — once a
    source topic is "spent" justifying one target topic, it's removed
    from the pool for every other course's evaluation.

    Args:
        courses: [{ "course_id": Any, "target_topics": [{"text", "embedding"}, ...] }, ...]
        pooled_topics: [{ "text": str, "embedding": list[float] }, ...]
            — the student's full pooled source topics, SHARED across
            all courses in this allocation (this is what makes it
            "cross-course" rather than per-course).
        topic_threshold: minimum similarity for a candidate match to be
            eligible for allocation at all. Default 0.75.

    Returns:
        { course_id: (per_topic_results, coverage_pct), ... }
        — same per-course shape as match_course(), but note: an
        unallocated target topic here shows best_match_text=None even
        if a nearby source topic exists in the pool, because that
        source topic may have been claimed by a different course's
        evaluation. This is the key behavioral difference from
        match_course(), which always shows the single nearest match
        regardless of what else is competing for it.

    Raises:
        ValueError: if courses is empty.
    """
    if not courses:
        raise ValueError("courses is empty — nothing to allocate.")

    # Build the global candidate list: every (course, target topic,
    # source topic) combination with its similarity, across ALL courses
    # at once — this is what lets allocation see conflicts between
    # courses, not just within one.
    candidates = []
    for course in courses:
        course_id = course["course_id"]
        for t_idx, target in enumerate(course["target_topics"]):
            for s_idx, source in enumerate(pooled_topics):
                sim = _cosine_similarity(target["embedding"], source["embedding"])
                if sim >= topic_threshold:
                    candidates.append((sim, course_id, t_idx, s_idx))

    # Highest similarity gets first claim on a source topic.
    candidates.sort(key=lambda c: c[0], reverse=True)

    claimed_targets = set()   # (course_id, t_idx) already allocated
    spent_sources = set()     # s_idx already used by some course

    # allocations[course_id][t_idx] = {"similarity": float, "source_text": str}
    allocations: dict = {course["course_id"]: {} for course in courses}

    for sim, course_id, t_idx, s_idx in candidates:
        if (course_id, t_idx) in claimed_targets:
            continue
        if s_idx in spent_sources:
            continue

        allocations[course_id][t_idx] = {
            "similarity": round(sim, 4),
            "source_text": pooled_topics[s_idx]["text"],
        }
        claimed_targets.add((course_id, t_idx))
        spent_sources.add(s_idx)

    # Assemble final per-course results in match_course()'s output shape.
    results_by_course = {}
    for course in courses:
        course_id = course["course_id"]
        target_topics = course["target_topics"]
        per_topic_results = []

        for t_idx, target in enumerate(target_topics):
            allocation = allocations[course_id].get(t_idx)
            if allocation is not None:
                per_topic_results.append({
                    "target_text": target["text"],
                    "best_match_text": allocation["source_text"],
                    "similarity": allocation["similarity"],
                    "covered": True,
                })
            else:
                per_topic_results.append({
                    "target_text": target["text"],
                    "best_match_text": None,
                    "similarity": 0.0,
                    "covered": False,
                })

        covered_count = sum(1 for r in per_topic_results if r["covered"])
        coverage_pct = round((covered_count / len(target_topics)) * 100, 2) if target_topics else 0.0
        results_by_course[course_id] = (per_topic_results, coverage_pct)

    return results_by_course


# ---------------------------------------------------------------------
# Standalone test — uses the same placeholder samples from the
# embeddings/ scripts so this is runnable without a DB or real data yet.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from embeddings.embed_rec_syllabus import SAMPLE_REC_SYLLABUS
    from embeddings.embed_source_syllabus import embed_source_topics
    from embeddings.embed_topics import get_embeddings_batch

    # Build target_topics from the REC placeholder syllabus (course_id=101)
    rec_topics = SAMPLE_REC_SYLLABUS[0]["topics"]
    rec_texts = [t["text"] for t in rec_topics]
    rec_embeddings = get_embeddings_batch(rec_texts)
    target_topics = [
        {"text": text, "embedding": emb}
        for text, emb in zip(rec_texts, rec_embeddings)
    ]

    # Student's pooled source topics — deliberately worded differently to
    # prove semantic (not literal) matching, plus one deliberately unrelated
    # topic that should NOT match anything, to test the threshold.
    student_topics_text = [
        "Deep learning fundamentals: perceptrons, forward and backward propagation",
        "Tree-based classifiers and their use in categorization problems",
        "Introduction to database normalization and SQL joins",  # unrelated
    ]
    pooled_topics = embed_source_topics(student_topics_text)

    results, coverage = match_course(target_topics, pooled_topics, topic_threshold=0.75)

    print(f"Coverage: {coverage}%\n")
    for r in results:
        status = "COVERED" if r["covered"] else "NOT COVERED"
        print(f"[{status}] (sim={r['similarity']})")
        print(f"  Target : {r['target_text']}")
        print(f"  Best match: {r['best_match_text']}\n")
