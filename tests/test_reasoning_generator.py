"""
tests/test_reasoning_generator.py
------------------------------------
Unit tests for reasoning_generator.py.

Tests run entirely offline: no GEMINI_API_KEY needed. generate() itself
uses the rule-based fallback automatically when no key is set (same
behavior as production would have before the key is configured), and
the JSON-parsing / prompt-building helpers are tested directly with
crafted inputs — no live API call involved anywhere here.

Run with:
    pytest tests/test_reasoning_generator.py -v
"""

import pytest
from reasoning_generator import (
    generate,
    _sanitize_match_results,
    _compute_deterministic_fields,
    _parse_gemini_json,
    _fallback_reasoning,
)


def make_match_results():
    return [
        {"target_text": "Unit 1: Linear regression", "best_match_text": "Regression basics",
         "similarity": 0.81, "covered": True},
        {"target_text": "Unit 2: Neural networks", "best_match_text": None,
         "similarity": 0.30, "covered": False},
    ]


class TestSanitizeMatchResults:
    def test_strips_student_identifying_fields(self):
        dirty = [{
            "target_text": "Unit 1",
            "best_match_text": "Match A",
            "similarity": 0.9,
            "covered": True,
            "student_name": "John Doe",     # should NOT survive sanitization
            "student_id": "REC12345",       # should NOT survive sanitization
        }]
        clean = _sanitize_match_results(dirty)

        assert "student_name" not in clean[0]
        assert "student_id" not in clean[0]
        assert clean[0] == {
            "target_text": "Unit 1",
            "best_match_text": "Match A",
            "similarity": 0.9,
            "covered": True,
        }

    def test_handles_missing_optional_fields(self):
        sparse = [{"target_text": "Unit 1", "covered": False}]
        clean = _sanitize_match_results(sparse)
        assert clean[0] == {"target_text": "Unit 1", "covered": False}


class TestDeterministicFields:
    def test_missing_units_extracted_correctly(self):
        result = _compute_deterministic_fields(make_match_results(), 50.0, None, None)
        assert result["missing_units"] == ["Unit 2: Neural networks"]

    def test_overlap_pct_passes_through_unchanged(self):
        result = _compute_deterministic_fields(make_match_results(), 73.5, None, None)
        assert result["overlap_pct"] == 73.5

    def test_credit_difference_computed_when_both_given(self):
        result = _compute_deterministic_fields(make_match_results(), 50.0, 4, 3)
        assert result["credit_difference"] == 1

    def test_credit_difference_none_when_not_given(self):
        result = _compute_deterministic_fields(make_match_results(), 50.0, None, None)
        assert result["credit_difference"] is None

    def test_no_missing_units_when_all_covered(self):
        all_covered = [{"target_text": "Unit 1", "covered": True}]
        result = _compute_deterministic_fields(all_covered, 100.0, None, None)
        assert result["missing_units"] == []


class TestParseGeminiJson:
    def test_parses_plain_json(self):
        text = '{"narrative": "Looks good", "learning_outcomes_match": "full", "suggested_bridge_module": null}'
        parsed = _parse_gemini_json(text)
        assert parsed["narrative"] == "Looks good"

    def test_strips_markdown_fences(self):
        text = '```json\n{"narrative": "Looks good", "learning_outcomes_match": "full"}\n```'
        parsed = _parse_gemini_json(text)
        assert parsed["learning_outcomes_match"] == "full"

    def test_strips_fences_without_language_tag(self):
        text = '```\n{"narrative": "OK"}\n```'
        parsed = _parse_gemini_json(text)
        assert parsed["narrative"] == "OK"

    def test_malformed_json_raises(self):
        with pytest.raises(Exception):
            _parse_gemini_json("not valid json at all")


class TestFallbackReasoning:
    def test_accept_maps_to_full(self):
        result = _fallback_reasoning(100.0, "accept", [])
        assert result["learning_outcomes_match"] == "full"
        assert result["suggested_bridge_module"] is None

    def test_partial_maps_to_partial_with_bridge_suggestion(self):
        result = _fallback_reasoning(65.0, "partial", ["Unit 2: Neural networks"])
        assert result["learning_outcomes_match"] == "partial"
        assert result["suggested_bridge_module"] is not None
        assert "Unit 2" in result["suggested_bridge_module"]

    def test_redo_maps_to_none(self):
        result = _fallback_reasoning(10.0, "redo", ["Unit 1", "Unit 2"])
        assert result["learning_outcomes_match"] == "none"

    def test_narrative_mentions_coverage_and_status(self):
        result = _fallback_reasoning(65.0, "partial", ["Unit 2"])
        assert "65.0" in result["narrative"]
        assert "partial" in result["narrative"]


class TestBridgeModuleGapTooLarge:
    """Aug 14 edge case: 'no bridge module possible' — when too many
    topics are missing, no single module should be suggested."""

    def test_small_gap_still_suggests_bridge(self):
        # 2 missing topics = at the threshold, bridge should still be suggested.
        result = _fallback_reasoning(60.0, "partial", ["Unit A", "Unit B"])
        assert result["suggested_bridge_module"] is not None

    def test_large_gap_suggests_no_bridge(self):
        # 5 missing topics — too broad for one module.
        missing = ["Unit A", "Unit B", "Unit C", "Unit D", "Unit E"]
        result = _fallback_reasoning(30.0, "partial", missing)
        assert result["suggested_bridge_module"] is None

    def test_large_gap_narrative_explains_why(self):
        missing = ["Unit A", "Unit B", "Unit C", "Unit D"]
        result = _fallback_reasoning(20.0, "partial", missing)
        assert "too many topics" in result["narrative"] or "single bridge module" in result["narrative"]


class TestVeryLowCoverageHonesty:
    """Aug 14 edge case: very low coverage should read as low, not be
    softened into sounding like partial credit was earned."""

    def test_near_zero_coverage_maps_to_redo_not_partial(self):
        # Confirms decision_engine + reasoning_generator agree: very low
        # coverage should never present as "partial" learning outcomes.
        result = _fallback_reasoning(2.0, "redo", ["Unit A", "Unit B", "Unit C"])
        assert result["learning_outcomes_match"] == "none"
        assert result["suggested_bridge_module"] is None

    def test_narrative_mentions_low_coverage_number(self):
        result = _fallback_reasoning(2.0, "redo", ["Unit A"])
        assert "2.0" in result["narrative"]


class TestMissingLearningOutcomesEntirely:
    """Aug 14 edge case: zero topics matched at all — reasoning must stay
    honest rather than fabricating partial overlap."""

    def test_zero_coverage_all_units_missing(self):
        all_missing = ["Unit 1", "Unit 2", "Unit 3"]
        result = generate(
            [{"target_text": u, "best_match_text": None, "similarity": 0.0, "covered": False}
             for u in all_missing],
            coverage_pct=0.0,
            recommended_status="redo",
        )
        assert result["overlap_pct"] == 0.0
        assert result["missing_units"] == all_missing
        assert result["learning_outcomes_match"] == "none"
        assert result["suggested_bridge_module"] is None


class TestGenerateWideGapEndToEnd:
    """Confirms the full generate() call (not just the fallback helper
    directly) respects the new bridge-module gap threshold."""

    def test_wide_gap_through_full_generate_call(self):
        wide_gap_results = [
            {"target_text": f"Unit {i}", "best_match_text": None, "similarity": 0.1, "covered": False}
            for i in range(5)
        ]
        result = generate(wide_gap_results, coverage_pct=15.0, recommended_status="partial")
        assert result["suggested_bridge_module"] is None


class TestGenerateEndToEnd:
    """Full generate() calls — run with no GEMINI_API_KEY set, so these
    exercise the real fallback path end-to-end, matching how the system
    behaves today before a key is configured."""

    def test_returns_all_required_keys(self):
        result = generate(make_match_results(), 50.0, "partial")
        expected_keys = {
            "overlap_pct", "learning_outcomes_match", "missing_units",
            "credit_difference", "suggested_bridge_module", "narrative",
        }
        assert set(result.keys()) == expected_keys

    def test_overlap_pct_matches_input(self):
        result = generate(make_match_results(), 50.0, "partial")
        assert result["overlap_pct"] == 50.0

    def test_missing_units_matches_uncovered_topics(self):
        result = generate(make_match_results(), 50.0, "partial")
        assert result["missing_units"] == ["Unit 2: Neural networks"]

    def test_credit_difference_populated_when_provided(self):
        result = generate(
            make_match_results(), 50.0, "partial",
            credit_hours_required=4, credit_hours_completed=2,
        )
        assert result["credit_difference"] == 2

    def test_full_coverage_no_missing_units(self):
        all_covered = [{"target_text": "Unit 1", "best_match_text": "Match",
                         "similarity": 0.9, "covered": True}]
        result = generate(all_covered, 100.0, "accept")
        assert result["missing_units"] == []
        assert result["learning_outcomes_match"] == "full"