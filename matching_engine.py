"""
matching_engine.py
--------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Compare a target course's topics (REC's requirements) against a
    student's pooled source-syllabus topics and produce:

        1. Per-topic semantic matches
        2. Overall coverage percentage
        3. Cross-course allocation without double-counting

Important allocation rule:

    A source topic MAY support multiple target topics inside the SAME
    REC target course.

    A source topic MUST NOT be reused to justify two DIFFERENT
    REC target courses.

Example:

    Source topic:
        "Arrays, strings, pointers and functions"

    It may support:

        REC Programming Using C:
            - Arrays
            - Strings
            - Pointers
            - Functions

    But once that source topic belongs to Programming Using C,
    it cannot also independently justify another REC course.
"""

import numpy as np


# ---------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------

def _cosine_similarity(
    vec_a: list[float],
    vec_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    MiniLM embeddings may already be normalized, but cosine similarity
    is calculated explicitly here so this function remains safe even
    if the embedding implementation changes later.
    """

    a = np.array(
        vec_a,
        dtype=float,
    )

    b = np.array(
        vec_b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


# ---------------------------------------------------------------------
# Single-course matching
# ---------------------------------------------------------------------

def match_course(
    target_topics: list[dict],
    pooled_topics: list[dict],
    topic_threshold: float = 0.75,
) -> tuple[list[dict], float]:
    """
    Match one REC target course against pooled source syllabus topics.

    Args:
        target_topics:
            [
                {
                    "text": str,
                    "embedding": list[float]
                },
                ...
            ]

        pooled_topics:
            [
                {
                    "text": str,
                    "embedding": list[float]
                },
                ...
            ]

        topic_threshold:
            Minimum cosine similarity required for a target topic
            to count as covered.

    Returns:
        (
            per_topic_results,
            coverage_pct
        )

    Example per-topic result:

        {
            "target_text": "Pointers",
            "best_match_text":
                "Pointers and Dynamic Memory Allocation",
            "similarity": 0.82,
            "covered": True
        }
    """

    if not target_topics:
        raise ValueError(
            "target_topics is empty — nothing to evaluate against."
        )

    if not 0.0 <= topic_threshold <= 1.0:
        raise ValueError(
            "topic_threshold must be between 0 and 1."
        )

    per_topic_results = []

    # -------------------------------------------------------------
    # No source topics submitted
    # -------------------------------------------------------------

    if not pooled_topics:

        for target in target_topics:

            per_topic_results.append(
                {
                    "target_text":
                        target["text"],
                    "best_match_text":
                        None,
                    "similarity":
                        0.0,
                    "covered":
                        False,
                }
            )

        return (
            per_topic_results,
            0.0,
        )

    # -------------------------------------------------------------
    # Match every REC target topic against every source topic
    # -------------------------------------------------------------

    for target in target_topics:

        best_similarity = -1.0
        best_match_text = None

        for source in pooled_topics:

            similarity = _cosine_similarity(
                target["embedding"],
                source["embedding"],
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_match_text = source["text"]

        covered = (
            best_similarity
            >= topic_threshold
        )

        per_topic_results.append(
            {
                "target_text":
                    target["text"],
                "best_match_text":
                    best_match_text,
                "similarity":
                    round(
                        best_similarity,
                        4,
                    ),
                "covered":
                    covered,
            }
        )

    # -------------------------------------------------------------
    # Coverage percentage
    # -------------------------------------------------------------

    covered_count = sum(
        1
        for result
        in per_topic_results
        if result["covered"]
    )

    coverage_pct = round(
        (
            covered_count
            / len(target_topics)
        )
        * 100,
        2,
    )

    return (
        per_topic_results,
        coverage_pct,
    )


# ---------------------------------------------------------------------
# Cross-course allocation
# ---------------------------------------------------------------------

def allocate_across_courses(
    courses: list[dict],
    pooled_topics: list[dict],
    topic_threshold: float = 0.75,
) -> dict:
    """
    Allocate source topics across multiple REC target courses.

    Important rule:

        A source topic may support multiple target topics within the
        SAME target course.

        A source topic may NOT support two DIFFERENT target courses.

    Allocation strategy:

        1. Compare every target topic against every source topic.
        2. Keep only similarities >= topic_threshold.
        3. Sort all candidate matches from strongest to weakest.
        4. Strongest claim gets first priority.
        5. Once a source topic belongs to one REC course, another
           REC course cannot use that same source topic.
        6. The same source topic may still support additional target
           topics in the course that already owns it.

    Returns:

        {
            course_id: (
                per_topic_results,
                coverage_pct
            )
        }
    """

    if not courses:
        raise ValueError(
            "courses is empty — nothing to allocate."
        )

    if not 0.0 <= topic_threshold <= 1.0:
        raise ValueError(
            "topic_threshold must be between 0 and 1."
        )

    # -------------------------------------------------------------
    # No source topics
    # -------------------------------------------------------------

    if not pooled_topics:

        results_by_course = {}

        for course in courses:

            course_id = course["course_id"]

            target_topics = course.get(
                "target_topics",
                [],
            )

            per_topic_results = []

            for target in target_topics:

                per_topic_results.append(
                    {
                        "target_text":
                            target["text"],
                        "best_match_text":
                            None,
                        "similarity":
                            0.0,
                        "covered":
                            False,
                    }
                )

            results_by_course[
                course_id
            ] = (
                per_topic_results,
                0.0,
            )

        return results_by_course

    # -------------------------------------------------------------
    # Candidate generation
    # -------------------------------------------------------------

    candidates = []

    for course in courses:

        course_id = course["course_id"]

        target_topics = course.get(
            "target_topics",
            [],
        )

        for target_index, target in enumerate(
            target_topics
        ):

            for source_index, source in enumerate(
                pooled_topics
            ):

                similarity = _cosine_similarity(
                    target["embedding"],
                    source["embedding"],
                )

                # Only threshold-qualified matches are candidates
                # for actual credit allocation.
                if similarity >= topic_threshold:

                    candidates.append(
                        (
                            similarity,
                            course_id,
                            target_index,
                            source_index,
                        )
                    )

    # -------------------------------------------------------------
    # Strongest semantic matches get first priority
    # -------------------------------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # A target topic should only receive one allocation.
    claimed_targets = set()

    # Maps:
    #
    # source_index -> REC course_id
    #
    # Once a source belongs to a course, other REC courses cannot
    # reuse it.
    #
    # But the SAME course can reuse it for multiple target topics.
    source_owner = {}

    # Stores successful allocations.
    #
    # {
    #     course_id: {
    #         target_index: {
    #             "similarity": ...,
    #             "source_text": ...
    #         }
    #     }
    # }
    allocations = {
        course["course_id"]: {}
        for course in courses
    }

    # -------------------------------------------------------------
    # Candidate allocation
    # -------------------------------------------------------------

    for (
        similarity,
        course_id,
        target_index,
        source_index,
    ) in candidates:

        target_key = (
            course_id,
            target_index,
        )

        # This REC target topic already has its strongest
        # permitted allocation.
        if target_key in claimed_targets:
            continue

        owner = source_owner.get(
            source_index
        )

        # ---------------------------------------------------------
        # Source belongs to another REC course
        # ---------------------------------------------------------

        if (
            owner is not None
            and owner != course_id
        ):
            continue

        # ---------------------------------------------------------
        # Allocate
        # ---------------------------------------------------------

        allocations[
            course_id
        ][
            target_index
        ] = {
            "similarity":
                round(
                    similarity,
                    4,
                ),
            "source_text":
                pooled_topics[
                    source_index
                ]["text"],
        }

        claimed_targets.add(
            target_key
        )

        # ---------------------------------------------------------
        # First REC course to use this source becomes its owner.
        #
        # The owner course may continue using this same source for
        # additional target topics.
        # ---------------------------------------------------------

        if owner is None:

            source_owner[
                source_index
            ] = course_id

    # -------------------------------------------------------------
    # Build final results
    # -------------------------------------------------------------

    results_by_course = {}

    for course in courses:

        course_id = course["course_id"]

        target_topics = course.get(
            "target_topics",
            [],
        )

        per_topic_results = []

        for target_index, target in enumerate(
            target_topics
        ):

            allocation = allocations[
                course_id
            ].get(
                target_index
            )

            # -----------------------------------------------------
            # Successfully allocated
            # -----------------------------------------------------

            if allocation is not None:

                per_topic_results.append(
                    {
                        "target_text":
                            target["text"],
                        "best_match_text":
                            allocation[
                                "source_text"
                            ],
                        "similarity":
                            allocation[
                                "similarity"
                            ],
                        "covered":
                            True,
                    }
                )

            # -----------------------------------------------------
            # Not allocated
            #
            # IMPORTANT:
            # Keep None / 0 here.
            #
            # A semantic match may have existed in isolation but
            # could have lost ownership to another target course.
            #
            # Returning that source text here would incorrectly make
            # it appear allocated and would break the no-double-
            # counting contract.
            # -----------------------------------------------------

            else:

                per_topic_results.append(
                    {
                        "target_text":
                            target["text"],
                        "best_match_text":
                            None,
                        "similarity":
                            0.0,
                        "covered":
                            False,
                    }
                )

        # ---------------------------------------------------------
        # Calculate coverage
        # ---------------------------------------------------------

        covered_count = sum(
            1
            for result
            in per_topic_results
            if result["covered"]
        )

        if target_topics:

            coverage_pct = round(
                (
                    covered_count
                    / len(target_topics)
                )
                * 100,
                2,
            )

        else:

            coverage_pct = 0.0

        results_by_course[
            course_id
        ] = (
            per_topic_results,
            coverage_pct,
        )

    return results_by_course


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from src.embeddings.embed_rec_syllabus import (
        SAMPLE_REC_SYLLABUS,
    )

    from src.embeddings.embed_source_syllabus import (
        embed_source_topics,
    )

    from src.embeddings.embed_topics import (
        get_embeddings_batch,
    )

    # -------------------------------------------------------------
    # Build sample REC target topics
    # -------------------------------------------------------------

    rec_topics = (
        SAMPLE_REC_SYLLABUS[0]["topics"]
    )

    rec_texts = [
        topic["text"]
        for topic in rec_topics
    ]

    rec_embeddings = get_embeddings_batch(
        rec_texts
    )

    target_topics = [
        {
            "text": text,
            "embedding": embedding,
        }
        for text, embedding
        in zip(
            rec_texts,
            rec_embeddings,
        )
    ]

    # -------------------------------------------------------------
    # Build sample student/source topics
    # -------------------------------------------------------------

    student_topics_text = [
        (
            "Deep learning fundamentals: "
            "perceptrons, forward and backward propagation"
        ),
        (
            "Tree-based classifiers and their use "
            "in categorization problems"
        ),
        (
            "Introduction to database normalization "
            "and SQL joins"
        ),
    ]

    pooled_topics = embed_source_topics(
        student_topics_text
    )

    # -------------------------------------------------------------
    # Independent single-course matching test
    # -------------------------------------------------------------

    results, coverage = match_course(
        target_topics=target_topics,
        pooled_topics=pooled_topics,
        topic_threshold=0.75,
    )

    print(
        f"\nCoverage: {coverage}%\n"
    )

    for result in results:

        status = (
            "COVERED"
            if result["covered"]
            else "NOT COVERED"
        )

        print(
            f"[{status}] "
            f"(sim={result['similarity']})"
        )

        print(
            "  Target     :",
            result["target_text"],
        )

        print(
            "  Best match :",
            result["best_match_text"],
        )

        print()