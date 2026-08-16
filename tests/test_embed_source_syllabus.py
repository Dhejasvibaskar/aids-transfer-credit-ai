"""
tests/test_embed_source_syllabus.py
--------------------------------------
Unit tests for embeddings/embed_source_syllabus.py.

The embedding model call is mocked (deterministic, offline, instant),
same pattern as the rest of this suite.

Run with:
    pytest tests/test_embed_source_syllabus.py -v
"""

import pytest

import embeddings.embed_source_syllabus as embed_source_syllabus
from embeddings.embed_source_syllabus import embed_source_topics


def fake_get_embeddings_batch(texts):
    """Deterministic stand-in for the real model — index-based fake
    vectors so we can confirm order is preserved, not just count."""
    return [[float(i), 0.0, 0.0] for i in range(len(texts))]


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    monkeypatch.setattr(embed_source_syllabus, "get_embeddings_batch", fake_get_embeddings_batch)


class TestEmbedSourceTopics:
    def test_returns_correct_shape(self):
        result = embed_source_topics(["topic one", "topic two"])

        assert len(result) == 2
        for item in result:
            assert set(item.keys()) == {"text", "embedding"}

    def test_preserves_input_order(self):
        result = embed_source_topics(["first", "second", "third"])

        assert result[0]["text"] == "first"
        assert result[0]["embedding"] == [0.0, 0.0, 0.0]
        assert result[1]["text"] == "second"
        assert result[1]["embedding"] == [1.0, 0.0, 0.0]
        assert result[2]["text"] == "third"
        assert result[2]["embedding"] == [2.0, 0.0, 0.0]

    def test_single_topic(self):
        result = embed_source_topics(["only topic"])
        assert len(result) == 1
        assert result[0]["text"] == "only topic"

    def test_empty_list_raises(self):
        # Matches the documented contract — embed_source_topics() is
        # meant to reject an empty submission explicitly rather than
        # silently returning [], since an empty transcript should be
        # caught and handled deliberately upstream, not embedded away.
        with pytest.raises(ValueError):
            embed_source_topics([])

    def test_output_feeds_match_course_shape(self):
        """Confirms the output shape is exactly what
        matching_engine.match_course()'s pooled_topics argument expects
        — {"text": str, "embedding": list[float]}."""
        result = embed_source_topics(["some topic"])
        item = result[0]
        assert isinstance(item["text"], str)
        assert isinstance(item["embedding"], list)
        assert all(isinstance(x, float) for x in item["embedding"])
