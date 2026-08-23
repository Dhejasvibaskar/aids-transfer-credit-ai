"""
embed_source_syllabus.py
--------------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    PER-EVALUATION embedding for whatever a transferring student submits —
    the "source" side (their old institution's course content). Unlike
    embed_rec_syllabus.py, this runs fresh every time a new transcript
    comes in, on whatever topics were extracted from that upload.

    In the live pipeline, this text comes from Person A's transcript/
    syllabus parsing step (POST /evaluations/upload). Today, before that
    parsing exists, you can pass in placeholder/custom topic text
    directly to test the embedding step in isolation.

    These embeddings are typically NOT written permanently to the shared
    `topic` table the way REC's syllabus is — they're transient, scoped
    to one evaluation. They get held in memory and passed straight into
    matching_engine.match_course().

Usage as a script (placeholder/manual test):
    python3 embed_source_syllabus.py

Usage as a module (real pipeline, once wired up):
    from embed_source_syllabus import embed_source_topics
    pooled = embed_source_topics(["Deep learning fundamentals...", ...])
"""

from src.embeddings.embed_topics import get_embeddings_batch


def embed_source_topics(topic_texts: list[str]) -> list[dict]:
    """
    Embed a student's submitted/pooled source topics for one evaluation.

    Args:
        topic_texts: raw topic text extracted from the student's
                      transcript/syllabus upload (or, for now, typed/
                      pasted in manually as placeholder input).

    Returns:
        [{ "text": str, "embedding": list[float] }, ...]
        — this is exactly the shape match_course() expects for its
        `pooled_topics` argument, per the shared interface contract.
    """
    if not topic_texts:
        raise ValueError("No source topics provided to embed.")

    if any(not text or not text.strip() for text in topic_texts):
        raise ValueError("Source topics cannot contain empty text.")

    embeddings = get_embeddings_batch(topic_texts)
    return [
        {"text": text, "embedding": embedding}
        for text, embedding in zip(topic_texts, embeddings)
    ]


# ---------------------------------------------------------------------
# Standalone test — customize SAMPLE_SOURCE_TOPICS below with any real
# syllabus text you have, or leave as-is for a quick placeholder check.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # EDIT THIS: paste in real source-syllabus topic text here to test
    # against something concrete, instead of the placeholder below.
    SAMPLE_SOURCE_TOPICS = [
        "Deep learning fundamentals: perceptrons, forward and backward propagation",
        "Tree-based classifiers and their use in categorization problems",
        "Introduction to regression analysis and error minimization",
    ]

    pooled = embed_source_topics(SAMPLE_SOURCE_TOPICS)

    print(f"Embedded {len(pooled)} source topics.\n")
    for item in pooled:
        print(f"Text: {item['text']}")
        print(f"Embedding length: {len(item['embedding'])}")
        print(f"First 5 values: {item['embedding'][:5]}\n")
