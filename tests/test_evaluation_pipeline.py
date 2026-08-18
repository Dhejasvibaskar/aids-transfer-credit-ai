"""
tests/test_evaluation_pipeline.py
------------------------------------
Integration tests for evaluation_pipeline.py — proves the full
embed -> match -> allocate -> decide -> reason chain works together,
including the Aug 13 edge cases (empty transcript, zero overlap,
single-topic course).

The embedding step (get_embeddings_batch) is mocked with hand-crafted
deterministic vectors so these tests run instantly and offline — same
reasoning as test_matching_engine.py. Everything AFTER embedding
(matching, allocation, decision, reasoning) runs for real, unmocked.

Run with:
    pytest tests/test_evaluation_pipeline.py -v
"""

import pytest
import evaluation_pipeline as pipeline

# Hand-crafted vectors, same convention as test_matching_engine.py.
IDENTICAL_A = [1.0, 0.0, 0.0]
IDENTICAL_B = [1.0, 0.0, 0.0]
ORTHOGONAL = [0.0, 1.0, 0.0]
PARTIAL = [0.7, 0.7, 0.0]

# Maps known mock topic text -> a controlled vector, so similarity
# outcomes are fully predictable in tests.
VECTOR_MAP = {
    "linear regression topic": IDENTICAL_A,
    "regression basics (matching source)": IDENTICAL_B,
    "unrelated database topic": ORTHOGONAL,
    "neural networks topic": PARTIAL,
}


def fake_get_embeddings_batch(texts):
    """Stand-in for the real model — looks up each text in VECTOR_MAP.
    Falls back to ORTHOGONAL (i.e. "matches nothing") for any text not
    explicitly mapped, so unmapped test text behaves predictably too."""
    return [VECTOR_MAP.get(t, ORTHOGONAL) for t in texts]


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Applied to every test in this file — replaces the real embedding
    call with the deterministic fake above."""
    monkeypatch.setattr(pipeline, "get_embeddings_batch", fake_get_embeddings_batch)


class TestSingleCourseFullPipeline:
    def test_full_match_end_to_end(self):
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic"],
            source_topic_texts=["regression basics (matching source)"],
        )
        assert result["coverage_pct"] == 100.0
        assert result["decision"]["status"] == "accept"
        assert result["decision"]["color"] == "green"
        assert result["reasoning"]["overlap_pct"] == 100.0
        assert result["reasoning"]["missing_units"] == []

    def test_partial_match_end_to_end(self):
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic", "unrelated database topic"],
            source_topic_texts=["regression basics (matching source)"],
        )
        assert result["coverage_pct"] == 50.0
        assert result["decision"]["status"] == "partial"
        assert result["decision"]["color"] == "yellow"
        assert len(result["reasoning"]["missing_units"]) == 1

    def test_zero_overlap_end_to_end(self):
        """Edge case: student's topics don't match the target course at all."""
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic"],
            source_topic_texts=["unrelated database topic"],
        )
        assert result["coverage_pct"] == 0.0
        assert result["decision"]["status"] == "redo"
        assert result["decision"]["color"] == "red"

    def test_empty_transcript_end_to_end(self):
        """Edge case: student submitted zero source topics."""
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic"],
            source_topic_texts=[],
        )
        assert result["coverage_pct"] == 0.0
        assert result["decision"]["status"] == "redo"
        assert result["match_results"][0]["best_match_text"] is None

    def test_single_topic_course_full_match(self):
        """Edge case: target course has only one topic total."""
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic"],
            source_topic_texts=["regression basics (matching source)"],
        )
        assert len(result["match_results"]) == 1
        assert result["coverage_pct"] == 100.0

    def test_credit_difference_flows_through(self):
        result = pipeline.run_single_course_evaluation(
            target_topic_texts=["linear regression topic"],
            source_topic_texts=["regression basics (matching source)"],
            credit_hours_required=4,
            credit_hours_completed=3,
        )
        assert result["reasoning"]["credit_difference"] == 1

    def test_empty_target_topics_raises(self):
        with pytest.raises(ValueError):
            pipeline.run_single_course_evaluation(
                target_topic_texts=[],
                source_topic_texts=["regression basics (matching source)"],
            )


class TestMultiCourseFullPipeline:
    def test_shared_source_topic_not_double_counted(self):
        """Confirms the full pipeline (not just allocate_across_courses in
        isolation) correctly prevents double-counting when wired through
        decision + reasoning too."""
        courses = [
            {"course_id": "ML101", "target_topic_texts": ["linear regression topic"]},
            {"course_id": "DM101", "target_topic_texts": ["neural networks topic"]},  # PARTIAL, weaker claim
        ]
        results = pipeline.run_multi_course_evaluation(
            courses,
            source_topic_texts=["regression basics (matching source)"],  # IDENTICAL_B — only one source topic
            topic_threshold=0.6,
        )

        # ML has the stronger match (IDENTICAL_A vs IDENTICAL_B = sim 1.0)
        assert results["ML101"]["decision"]["status"] == "accept"
        # DM's weaker claim loses the (only) source topic to ML
        assert results["DM101"]["decision"]["status"] == "redo"
        assert results["DM101"]["coverage_pct"] == 0.0

    def test_empty_courses_raises(self):
        with pytest.raises(ValueError):
            pipeline.run_multi_course_evaluation([], source_topic_texts=["x"])

    def test_course_with_no_target_topics_raises(self):
        courses = [{"course_id": "C1", "target_topic_texts": []}]
        with pytest.raises(ValueError):
            pipeline.run_multi_course_evaluation(courses, source_topic_texts=["x"])

    def test_each_course_gets_full_result_shape(self):
        courses = [{"course_id": "C1", "target_topic_texts": ["linear regression topic"]}]
        results = pipeline.run_multi_course_evaluation(
            courses, source_topic_texts=["regression basics (matching source)"]
        )
        assert set(results["C1"].keys()) == {"match_results", "coverage_pct", "decision", "reasoning"}
