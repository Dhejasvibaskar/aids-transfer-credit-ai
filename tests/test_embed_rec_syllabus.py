"""
tests/test_embed_rec_syllabus.py
-----------------------------------
Unit tests for embeddings/embed_rec_syllabus.py.

The embedding model call is mocked (deterministic, offline, instant) —
same reasoning as the other test files in this suite. The DB write path
is tested with a mocked psycopg2 connection/cursor, so no real database
is needed to verify the SQL gets called correctly.

Run with:
    pytest tests/test_embed_rec_syllabus.py -v
"""

import json
import pytest
from unittest.mock import MagicMock

import src.embeddings.embed_rec_syllabus as embed_rec_syllabus
from src.embeddings.embed_rec_syllabus import (
    load_syllabus,
    embed_rec_syllabus as embed_rec_syllabus_fn,
    validate_syllabus,
    SAMPLE_REC_SYLLABUS,
)


def fake_get_embeddings_batch(texts):
    """Deterministic stand-in for the real model — one fixed-length
    fake vector per text, so shape/count can be asserted reliably."""
    return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    monkeypatch.setattr(embed_rec_syllabus, "get_embeddings_batch", fake_get_embeddings_batch)


class TestLoadSyllabus:
    def test_no_path_returns_placeholder(self):
        result = load_syllabus(None)
        assert result == SAMPLE_REC_SYLLABUS

    def test_loads_real_json_file(self, tmp_path):
        custom_syllabus = [
            {"course_id": 999, "course_code": "TEST101",
             "topics": [{"unit_no": 1, "text": "Custom topic text"}]}
        ]
        file_path = tmp_path / "syllabus.json"
        file_path.write_text(json.dumps(custom_syllabus))

        result = load_syllabus(str(file_path))
        assert result == custom_syllabus

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_syllabus("/nonexistent/path/syllabus.json")


class TestValidateSyllabus:
    def test_valid_syllabus_passes(self):
        validate_syllabus(SAMPLE_REC_SYLLABUS)  # should not raise

    def test_empty_syllabus_raises(self):
        with pytest.raises(ValueError, match="Syllabus is empty"):
            validate_syllabus([])

    def test_missing_course_id_raises(self):
        bad = [{"topics": [{"unit_no": 1, "text": "x"}]}]
        with pytest.raises(ValueError, match="Missing course_id"):
            validate_syllabus(bad)

    def test_missing_topics_raises(self):
        bad = [{"course_id": 1, "topics": []}]
        with pytest.raises(ValueError, match="No topics found"):
            validate_syllabus(bad)

    def test_missing_unit_no_raises(self):
        bad = [{"course_id": 1, "topics": [{"text": "x"}]}]
        with pytest.raises(ValueError, match="Missing unit_no"):
            validate_syllabus(bad)

    def test_missing_topic_text_raises(self):
        bad = [{"course_id": 1, "topics": [{"unit_no": 1}]}]
        with pytest.raises(ValueError, match="Missing topic text"):
            validate_syllabus(bad)

    def test_blank_topic_text_raises(self):
        bad = [{"course_id": 1, "topics": [{"unit_no": 1, "text": "   "}]}]
        with pytest.raises(ValueError, match="Missing topic text"):
            validate_syllabus(bad)


class TestEmbedRecSyllabusDryRun:
    def test_dry_run_returns_correct_shape(self):
        results = embed_rec_syllabus_fn(SAMPLE_REC_SYLLABUS, dry_run=True)

        assert len(results) == 3  # 3 topics in the placeholder course
        for r in results:
            assert set(r.keys()) == {"course_id", "unit_no", "text", "embedding"}
            assert r["course_id"] == 101
            assert r["embedding"] == [0.1, 0.2, 0.3]

    def test_dry_run_does_not_touch_db(self):
        mock_conn = MagicMock()
        embed_rec_syllabus_fn(SAMPLE_REC_SYLLABUS, conn=mock_conn, dry_run=True)

        mock_conn.cursor.assert_not_called()
        mock_conn.commit.assert_not_called()

    def test_multiple_courses_all_processed(self):
        two_courses = [
            {"course_id": 1, "course_code": "A",
             "topics": [{"unit_no": 1, "text": "topic A1"}]},
            {"course_id": 2, "course_code": "B",
             "topics": [{"unit_no": 1, "text": "topic B1"}, {"unit_no": 2, "text": "topic B2"}]},
        ]
        results = embed_rec_syllabus_fn(two_courses, dry_run=True)

        assert len(results) == 3
        course_ids = {r["course_id"] for r in results}
        assert course_ids == {1, 2}

    def test_empty_syllabus_raises(self):
        # Updated: embed_rec_syllabus() now validates input up front and
        # rejects an empty syllabus explicitly, rather than silently
        # returning [].
        with pytest.raises(ValueError, match="Syllabus is empty"):
            embed_rec_syllabus_fn([], dry_run=True)


class TestEmbedRecSyllabusDbWrite:
    def test_writes_to_db_when_not_dry_run(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        embed_rec_syllabus_fn(SAMPLE_REC_SYLLABUS, conn=mock_conn, dry_run=False)

        # One execute() call per topic (3 topics in the placeholder).
        assert mock_cursor.execute.call_count == 3
        # Updated: commit() is now called ONCE after all topics are
        # written, not once per topic — a batching improvement over the
        # original per-topic-commit implementation.
        assert mock_conn.commit.call_count == 1

    def test_execute_called_with_correct_course_and_unit(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        single_topic_syllabus = [
            {"course_id": 555, "course_code": "X",
             "topics": [{"unit_no": 7, "text": "some topic"}]}
        ]
        embed_rec_syllabus_fn(single_topic_syllabus, conn=mock_conn, dry_run=False)

        call_args = mock_cursor.execute.call_args
        sql, params = call_args[0]
        assert "UPDATE topic" in sql
        assert params[1] == 555  # course_id
        assert params[2] == 7    # unit_no