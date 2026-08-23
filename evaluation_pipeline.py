"""
evaluation_pipeline.py
----------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Chains together embedding, matching, source-course allocation,
    decision, and reasoning.

Current production design:
    1. Keep source-course identity.
    2. Expand broad source syllabus units into smaller atomic chunks.
    3. Compare each completed source course with each REC target course.
    4. Apply hybrid topic coverage:
         - semantic similarity, plus
         - strong lexical/concept evidence.
    5. Gate source->target claims using course-title/subject compatibility.
    6. Prevent language-specific mismatches such as C -> Java.
    7. Protect strong direct course-identity matches first.
    8. Allocate each source course to only one REC target course.
    9. Allow multiple source courses to contribute to one REC target.
    10. Calculate final coverage and ACCEPT / PARTIAL / REDO.
"""

import re

from src.embeddings.embed_topics import get_embeddings_batch

from matching_engine import (
    match_course,
    allocate_across_courses,
)

from decision_engine import decide_with_color
from reasoning_generator import generate


# ---------------------------------------------------------------------
# Allocation settings
# ---------------------------------------------------------------------

DIRECT_TITLE_LOCK_THRESHOLD = 0.80


# ---------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------

def _embed_topic_texts(
    texts: list[str],
) -> list[dict]:
    """
    Convert raw topic strings into:

        {
            "text": ...,
            "embedding": [...]
        }

    Empty input returns [] without calling the model.
    """

    if not texts:
        return []

    embeddings = get_embeddings_batch(
        texts
    )

    return [
        {
            "text": text,
            "embedding": embedding,
        }
        for text, embedding in zip(
            texts,
            embeddings,
        )
    ]


# ---------------------------------------------------------------------
# Atomic topic expansion
# ---------------------------------------------------------------------

def _expand_atomic_topics(
    texts: list[str],
) -> list[str]:
    """
    Break broad source syllabus units into smaller semantic chunks.

    The original full unit is always retained.

    This helps when REC stores short target topics such as:
        "Pointers"
        "Character Arrays and Strings Handling Functions"

    while the source syllabus stores a long unit paragraph containing
    many concepts together.
    """

    expanded = []
    seen = set()

    def add_chunk(value):
        if not value:
            return

        value = " ".join(
            str(value).split()
        ).strip(
            " ,;:-"
        )

        if len(value) < 4:
            return

        key = value.lower()

        if key in seen:
            return

        seen.add(key)

        expanded.append(
            value
        )

    for text in texts or []:

        if not text:
            continue

        clean = " ".join(
            str(text).split()
        )

        # Keep original full unit.
        add_chunk(
            clean
        )

        # Split only dashes surrounded by spaces.
        major_parts = re.split(
            r"\s+[\u2013\u2014]\s+|\s+-\s+",
            clean,
        )

        for major_part in major_parts:

            major_part = (
                major_part.strip()
            )

            if not major_part:
                continue

            add_chunk(
                major_part
            )

            clauses = re.split(
                r"(?<=[.;])\s+|;",
                major_part,
            )

            for clause in clauses:

                clause = clause.strip(
                    " .;"
                )

                if not clause:
                    continue

                add_chunk(
                    clause
                )

                comma_items = [
                    item.strip(
                        " .;"
                    )
                    for item
                    in clause.split(",")
                    if item.strip(
                        " .;"
                    )
                ]

                if (
                    2
                    <= len(comma_items)
                    <= 20
                ):

                    cleaned_items = []

                    for item in comma_items:

                        item = re.sub(
                            r"^(and|or)\s+",
                            "",
                            item,
                            flags=re.IGNORECASE,
                        ).strip()

                        if not item:
                            continue

                        cleaned_items.append(
                            item
                        )

                        if len(item) >= 5:
                            add_chunk(
                                item
                            )

                    # Two-concept windows.
                    for index in range(
                        len(cleaned_items) - 1
                    ):
                        add_chunk(
                            cleaned_items[index]
                            + ", "
                            + cleaned_items[index + 1]
                        )

                    # Three-concept windows.
                    for index in range(
                        len(cleaned_items) - 2
                    ):
                        add_chunk(
                            cleaned_items[index]
                            + ", "
                            + cleaned_items[index + 1]
                            + ", "
                            + cleaned_items[index + 2]
                        )

    return expanded


# ---------------------------------------------------------------------
# Hybrid topic matching helpers
# ---------------------------------------------------------------------

_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "of",
    "to",
    "for",
    "with",
    "using",
    "use",
    "into",
    "from",
    "on",
    "in",
    "by",
    "as",
    "at",
    "is",
    "are",
    "be",
    "basic",
    "basics",
    "introduction",
    "overview",
    "unit",
    "concept",
    "concepts",
    "method",
    "methods",
}


_TOKEN_ALIASES = {
    "arrays": "array",
    "array": "array",

    "strings": "string",
    "string": "string",

    "pointers": "pointer",
    "pointer": "pointer",

    "functions": "function",
    "function": "function",

    "variables": "variable",
    "variable": "variable",

    "operators": "operator",
    "operator": "operator",

    "expressions": "expression",
    "expression": "expression",

    "structures": "structure",
    "structure": "structure",

    "loops": "loop",
    "looping": "loop",
    "loop": "loop",

    "recursive": "recurs",
    "recursion": "recurs",

    "preprocessors": "preprocessor",
    "preprocessor": "preprocessor",

    "decisions": "decision",
    "decision": "decision",

    "branches": "branch",
    "branching": "branch",

    "inputs": "input",
    "input": "input",

    "outputs": "output",
    "output": "output",

    "operations": "operation",
    "operation": "operation",

    "algorithms": "algorithm",
    "algorithm": "algorithm",
}


def _canonical_topic_token(
    token: str,
) -> str:

    token = (
        token
        .lower()
        .strip()
    )

    if not token:
        return ""

    if token in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[
            token
        ]

    # Light plural normalization.
    if (
        len(token) > 5
        and token.endswith(
            "ies"
        )
    ):
        token = (
            token[:-3]
            + "y"
        )

    elif (
        len(token) > 4
        and token.endswith(
            "s"
        )
        and not token.endswith(
            "ss"
        )
    ):
        token = token[:-1]

    return _TOKEN_ALIASES.get(
        token,
        token,
    )


def _topic_tokens(
    text: str,
) -> list[str]:

    raw_tokens = re.findall(
        r"[A-Za-z0-9]+",
        str(
            text or ""
        ).lower(),
    )

    tokens = []

    for token in raw_tokens:

        if token in _TOPIC_STOPWORDS:
            continue

        canonical = (
            _canonical_topic_token(
                token
            )
        )

        if not canonical:
            continue

        if canonical in _TOPIC_STOPWORDS:
            continue

        if len(canonical) == 1:
            continue

        tokens.append(
            canonical
        )

    return tokens


def _tokens_equivalent(
    left: str,
    right: str,
) -> bool:

    if left == right:
        return True

    if (
        len(left) >= 5
        and len(right) >= 5
    ):
        return (
            left[:5]
            == right[:5]
        )

    return False


def _lexical_topic_coverage(
    target_text: str,
    source_topics: list[dict],
) -> float:
    """
    Estimate how much of a REC target topic is explicitly present
    somewhere inside the same source course.
    """

    target_tokens = list(
        dict.fromkeys(
            _topic_tokens(
                target_text
            )
        )
    )

    if not target_tokens:
        return 0.0

    source_tokens = []

    for topic in source_topics or []:

        if isinstance(
            topic,
            dict,
        ):
            text = topic.get(
                "text",
                "",
            )

        else:
            text = str(
                topic
            )

        source_tokens.extend(
            _topic_tokens(
                text
            )
        )

    source_tokens = list(
        dict.fromkeys(
            source_tokens
        )
    )

    if not source_tokens:
        return 0.0

    matched = 0

    for target_token in target_tokens:

        if any(
            _tokens_equivalent(
                target_token,
                source_token,
            )
            for source_token
            in source_tokens
        ):
            matched += 1

    return (
        matched
        / len(target_tokens)
    )


def _apply_hybrid_topic_coverage(
    match_results: list[dict],
    source_topics: list[dict],
    topic_threshold: float,
) -> tuple[list[dict], float]:
    """
    Hybrid topic coverage rules:

        1. semantic >= topic_threshold
        2. lexical >= 0.80
        3. semantic >= 0.60 AND lexical >= 0.50
        4. semantic >= 0.55 AND lexical >= 0.65
    """

    adjusted_results = []

    for result in match_results or []:

        adjusted = dict(
            result
        )

        similarity = float(
            adjusted.get(
                "similarity",
                0.0,
            )
            or 0.0
        )

        lexical_coverage = (
            _lexical_topic_coverage(
                adjusted.get(
                    "target_text",
                    "",
                ),
                source_topics,
            )
        )

        covered = (
            similarity
            >= topic_threshold
        )

        coverage_method = (
            "semantic"
            if covered
            else None
        )

        if (
            not covered
            and lexical_coverage
            >= 0.80
        ):
            covered = True
            coverage_method = (
                "lexical"
            )

        elif (
            not covered
            and similarity >= 0.60
            and lexical_coverage
            >= 0.50
        ):
            covered = True
            coverage_method = (
                "hybrid"
            )

        elif (
            not covered
            and similarity >= 0.55
            and lexical_coverage
            >= 0.65
        ):
            covered = True
            coverage_method = (
                "hybrid"
            )

        adjusted[
            "covered"
        ] = covered

        adjusted[
            "lexical_coverage"
        ] = round(
            lexical_coverage,
            4,
        )

        adjusted[
            "coverage_method"
        ] = coverage_method

        adjusted_results.append(
            adjusted
        )

    if not adjusted_results:
        return (
            adjusted_results,
            0.0,
        )

    covered_count = sum(
        1
        for result in adjusted_results
        if result.get(
            "covered"
        )
    )

    coverage_pct = (
        covered_count
        / len(adjusted_results)
        * 100.0
    )

    return (
        adjusted_results,
        round(
            coverage_pct,
            2,
        ),
    )


# ---------------------------------------------------------------------
# Course-title / subject compatibility
# ---------------------------------------------------------------------

_TITLE_STOPWORDS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "using",
    "use",
    "lab",
    "laboratory",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
}


def _normalize_title_word(
    token: str,
) -> str:

    token = (
        token
        .lower()
        .strip()
    )

    aliases = {
        "foundations":
            "foundation",

        "principles":
            "principle",

        "techniques":
            "technique",

        "systems":
            "system",

        "applications":
            "application",

        "structures":
            "structure",

        "algorithms":
            "algorithm",

        "communications":
            "communication",

        "programming":
            "programming",
    }

    if token in aliases:
        return aliases[
            token
        ]

    if (
        len(token) > 5
        and token.endswith(
            "ies"
        )
    ):
        return (
            token[:-3]
            + "y"
        )

    if (
        len(token) > 4
        and token.endswith(
            "s"
        )
        and not token.endswith(
            "ss"
        )
    ):
        return token[:-1]

    return token


def _title_tokens(
    title: str,
) -> set[str]:

    clean = str(
        title or ""
    ).lower()

    clean = clean.replace(
        "c++",
        " cpp ",
    )

    clean = clean.replace(
        "c#",
        " csharp ",
    )

    raw_tokens = re.findall(
        r"[a-z0-9]+",
        clean,
    )

    tokens = set()

    for token in raw_tokens:

        if token in _TITLE_STOPWORDS:
            continue

        token = (
            _normalize_title_word(
                token
            )
        )

        if (
            not token
            or token
            in _TITLE_STOPWORDS
        ):
            continue

        tokens.add(
            token
        )

    return tokens


def _course_title_similarity(
    source_title: str,
    target_title: str,
) -> float:
    """
    Title overlap score based on the smaller title.
    """

    source_tokens = (
        _title_tokens(
            source_title
        )
    )

    target_tokens = (
        _title_tokens(
            target_title
        )
    )

    if (
        not source_tokens
        or not target_tokens
    ):
        return 0.0

    intersection = (
        source_tokens
        & target_tokens
    )

    return (
        len(intersection)
        / min(
            len(source_tokens),
            len(target_tokens),
        )
    )


def _extract_language_markers(
    text: str,
) -> set[str]:
    """
    Detect explicit programming-language identity.

    Prevents mappings such as:
        C Programming
            ->
        OOP Using Java
    """

    raw = (
        f" {str(text or '').lower()} "
    )

    raw = raw.replace(
        "c++",
        " cpp ",
    )

    raw = raw.replace(
        "c#",
        " csharp ",
    )

    markers = set()

    patterns = {
        "cpp":
            r"\bcpp\b",

        "csharp":
            r"\bcsharp\b",

        "java":
            r"\bjava\b",

        "python":
            r"\bpython\b",

        "javascript":
            r"\bjavascript\b|\bjava script\b",

        "typescript":
            r"\btypescript\b|\btype script\b",

        "r":
            r"\br programming\b",

        "c":
            (
                r"\bc programming\b|"
                r"\bprogramming in c\b|"
                r"\bprogramming using c\b"
            ),
    }

    for marker, pattern in (
        patterns.items()
    ):

        if re.search(
            pattern,
            raw,
        ):
            markers.add(
                marker
            )

    return markers


def _course_claim_allowed(
    source_title: str,
    source_topic_texts: list[str],
    target_title: str,
    pair_coverage: float,
    covered_count: int,
) -> tuple[bool, float, str]:
    """
    Decide whether a source course is academically plausible
    as a candidate for a REC target course.

    Returns:
        (
            allowed,
            title_score,
            reason
        )
    """

    title_score = (
        _course_title_similarity(
            source_title,
            target_title,
        )
    )

    source_language_text = " ".join(
        [
            source_title
        ]
        + list(
            source_topic_texts
            or []
        )
    )

    source_languages = (
        _extract_language_markers(
            source_language_text
        )
    )

    target_languages = (
        _extract_language_markers(
            target_title
        )
    )

    # Programming language target requires
    # matching language in source.
    if target_languages:

        if not source_languages:

            return (
                False,
                title_score,
                "target_language_missing_in_source",
            )

        if target_languages.isdisjoint(
            source_languages
        ):

            return (
                False,
                title_score,
                "programming_language_conflict",
            )

    # Normal title compatibility.
    if title_score >= 0.45:

        return (
            True,
            title_score,
            "title_compatible",
        )

    # Strong content override.
    if (
        pair_coverage >= 60.0
        and covered_count >= 2
    ):

        return (
            True,
            title_score,
            "strong_content_override",
        )

    return (
        False,
        title_score,
        "weak_course_identity",
    )


# ---------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------

def _average_similarity(
    match_results: list[dict],
) -> float:
    """
    Average best-match similarity for a course comparison.

    Used only as a tie-breaker / allocation signal.
    """

    if not match_results:
        return 0.0

    values = [
        float(
            result.get(
                "similarity",
                0.0,
            )
        )
        for result
        in match_results
    ]

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


# ---------------------------------------------------------------------
# Normalize source-course input
# ---------------------------------------------------------------------

def _normalize_source_courses(
    source_courses: list[dict],
) -> list[dict]:
    """
    Normalize source syllabus courses.
    """

    normalized = []

    for index, course in enumerate(
        source_courses or []
    ):

        if not isinstance(
            course,
            dict,
        ):
            continue

        course_id = (
            course.get(
                "source_course_id"
            )
            or course.get(
                "course_id"
            )
            or course.get(
                "code"
            )
            or f"source_{index}"
        )

        course_title = (
            course.get(
                "source_course_title"
            )
            or course.get(
                "course_title"
            )
            or course.get(
                "title"
            )
            or ""
        )

        topic_texts = (
            course.get(
                "topic_texts"
            )
            or course.get(
                "units"
            )
            or course.get(
                "topics"
            )
            or []
        )

        if isinstance(
            topic_texts,
            str,
        ):
            topic_texts = [
                topic_texts
            ]

        cleaned_topics = []

        for topic in topic_texts:

            if isinstance(
                topic,
                str,
            ):
                text = (
                    topic.strip()
                )

            elif isinstance(
                topic,
                dict,
            ):
                text = str(
                    topic.get(
                        "text"
                    )
                    or topic.get(
                        "title"
                    )
                    or topic.get(
                        "content"
                    )
                    or ""
                ).strip()

            else:
                text = ""

            if text:
                cleaned_topics.append(
                    text
                )

        if not cleaned_topics:
            continue

        normalized.append(
            {
                "source_course_id":
                    str(course_id),

                "source_course_title":
                    str(course_title),

                "topic_texts":
                    cleaned_topics,
            }
        )

    return normalized


# ---------------------------------------------------------------------
# Single-course evaluation
# ---------------------------------------------------------------------

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
    Full pipeline for ONE target course.

    Kept compatible with the earlier interface.
    """

    if not target_topic_texts:
        raise ValueError(
            "target_topic_texts is empty - "
            "target course has no topics to evaluate."
        )

    target_topics = (
        _embed_topic_texts(
            target_topic_texts
        )
    )

    pooled_topics = (
        _embed_topic_texts(
            source_topic_texts
        )
    )

    match_results, _ = (
        match_course(
            target_topics,
            pooled_topics,
            topic_threshold,
        )
    )

    (
        match_results,
        coverage_pct,
    ) = _apply_hybrid_topic_coverage(
        match_results,
        pooled_topics,
        topic_threshold,
    )

    decision = decide_with_color(
        coverage_pct,
        accept_threshold,
        partial_threshold,
    )

    reasoning = generate(
        match_results,
        coverage_pct,
        decision["status"],
        credit_hours_required=
            credit_hours_required,
        credit_hours_completed=
            credit_hours_completed,
    )

    return {
        "match_results":
            match_results,

        "coverage_pct":
            coverage_pct,

        "decision":
            decision,

        "reasoning":
            reasoning,
    }


# ---------------------------------------------------------------------
# Legacy pooled-topic multi-course implementation
# ---------------------------------------------------------------------

def _run_multi_course_flat_pool(
    courses: list[dict],
    source_topic_texts: list[str],
    topic_threshold: float,
    accept_threshold: float,
    partial_threshold: float,
) -> dict:
    """
    Original multi-course behaviour kept for compatibility
    with older tests/callers.
    """

    pooled_topics = (
        _embed_topic_texts(
            source_topic_texts
        )
    )

    courses_for_allocation = []

    for course in courses:

        if not course[
            "target_topic_texts"
        ]:
            raise ValueError(
                f"course_id="
                f"{course['course_id']} "
                "has no target topics."
            )

        courses_for_allocation.append(
            {
                "course_id":
                    course[
                        "course_id"
                    ],

                "target_topics":
                    _embed_topic_texts(
                        course[
                            "target_topic_texts"
                        ]
                    ),
            }
        )

    allocation_results = (
        allocate_across_courses(
            courses_for_allocation,
            pooled_topics,
            topic_threshold,
        )
    )

    final_results = {}

    for (
        course_id,
        (
            match_results,
            _,
        ),
    ) in allocation_results.items():

        (
            match_results,
            coverage_pct,
        ) = _apply_hybrid_topic_coverage(
            match_results,
            pooled_topics,
            topic_threshold,
        )

        decision = (
            decide_with_color(
                coverage_pct,
                accept_threshold,
                partial_threshold,
            )
        )

        reasoning = generate(
            match_results,
            coverage_pct,
            decision[
                "status"
            ],
        )

        final_results[
            course_id
        ] = {
            "match_results":
                match_results,

            "coverage_pct":
                coverage_pct,

            "decision":
                decision,

            "reasoning":
                reasoning,
        }

    return final_results


# ---------------------------------------------------------------------
# Structured source-course allocation
# ---------------------------------------------------------------------

def _run_multi_course_structured(
    courses: list[dict],
    source_courses: list[dict],
    topic_threshold: float,
    accept_threshold: float,
    partial_threshold: float,
) -> dict:
    """
    Production multi-course evaluation.

    Rules:
        - Obvious same-course matches are protected first.
        - One completed source course contributes to only one
          REC target course.
        - Multiple source courses may contribute to one REC target.
        - Course-title/subject compatibility prevents accidental claims.
        - Language-specific targets require the same language.
        - Final topic coverage uses semantic + lexical evidence.
    """

    normalized_sources = (
        _normalize_source_courses(
            source_courses
        )
    )

    # -------------------------------------------------------------
    # Embed REC target courses
    # -------------------------------------------------------------

    embedded_targets = {}

    for course in courses:

        course_id = course[
            "course_id"
        ]

        target_texts = (
            course.get(
                "target_topic_texts",
                [],
            )
        )

        if not target_texts:

            raise ValueError(
                f"course_id={course_id} "
                "has no target topics."
            )

        embedded_targets[
            course_id
        ] = _embed_topic_texts(
            target_texts
        )

    # -------------------------------------------------------------
    # Expand and embed each completed source course independently
    # -------------------------------------------------------------

    embedded_sources = {}
    source_metadata = {}

    for source_course in normalized_sources:

        source_id = source_course[
            "source_course_id"
        ]

        raw_topic_texts = (
            source_course[
                "topic_texts"
            ]
        )

        atomic_topic_texts = (
            _expand_atomic_topics(
                raw_topic_texts
            )
        )

        embedded_sources[
            source_id
        ] = _embed_topic_texts(
            atomic_topic_texts
        )

        source_metadata[
            source_id
        ] = {
            "title":
                source_course[
                    "source_course_title"
                ],

            "topic_texts":
                atomic_topic_texts,

            "original_topic_texts":
                raw_topic_texts,
        }

        print(
            f"ATOMIC TOPICS: "
            f"{source_id} - "
            f"{source_course['source_course_title']} "
            f"| original="
            f"{len(raw_topic_texts)} "
            f"| expanded="
            f"{len(atomic_topic_texts)}"
        )

    # -------------------------------------------------------------
    # No usable source courses
    # -------------------------------------------------------------

    if not embedded_sources:

        final_results = {}

        for course in courses:

            course_id = (
                course[
                    "course_id"
                ]
            )

            target_topics = (
                embedded_targets[
                    course_id
                ]
            )

            (
                match_results,
                coverage_pct,
            ) = match_course(
                target_topics,
                [],
                topic_threshold,
            )

            decision = (
                decide_with_color(
                    coverage_pct,
                    accept_threshold,
                    partial_threshold,
                )
            )

            reasoning = generate(
                match_results,
                coverage_pct,
                decision[
                    "status"
                ],
            )

            final_results[
                course_id
            ] = {
                "match_results":
                    match_results,

                "coverage_pct":
                    coverage_pct,

                "decision":
                    decision,

                "reasoning":
                    reasoning,

                "allocated_source_courses":
                    [],
            }

        return final_results

    # -------------------------------------------------------------
    # Build source-course -> REC target-course claims
    # -------------------------------------------------------------

    claims = []

    for course in courses:

        course_id = (
            course[
                "course_id"
            ]
        )

        target_topics = (
            embedded_targets[
                course_id
            ]
        )

        target_code = (
            course.get(
                "course_code",
                "",
            )
        )

        target_title = (
            course.get(
                "course_title",
                "",
            )
        )

        for (
            source_id,
            source_topics,
        ) in embedded_sources.items():

            (
                raw_pair_results,
                _,
            ) = match_course(
                target_topics,
                source_topics,
                topic_threshold,
            )

            (
                pair_match_results,
                pair_coverage,
            ) = _apply_hybrid_topic_coverage(
                raw_pair_results,
                source_topics,
                topic_threshold,
            )

            average_similarity = (
                _average_similarity(
                    pair_match_results
                )
            )

            covered_count = sum(
                1
                for result
                in pair_match_results
                if result.get(
                    "covered"
                )
            )

            if covered_count == 0:
                continue

            source_title = (
                source_metadata[
                    source_id
                ][
                    "title"
                ]
            )

            (
                allowed,
                title_score,
                gate_reason,
            ) = _course_claim_allowed(
                source_title=
                    source_title,

                source_topic_texts=
                    source_metadata[
                        source_id
                    ][
                        "topic_texts"
                    ],

                target_title=
                    target_title,

                pair_coverage=
                    pair_coverage,

                covered_count=
                    covered_count,
            )

            # -----------------------------------------------------
            # Focused debug pairs
            # -----------------------------------------------------

            debug_pair = (
    # Existing good/reference pairs
    (
        source_id == "25CS103"
        and target_code == "GE23131"
    )
    or
    (
        source_id == "25TA101"
        and target_code == "GE23117"
    )
    or
    (
        source_id == "25TA201"
        and target_code == "GE23217"
    )
    or
    (
        source_id == "25CSC01"
        and target_code == "CS23231"
    )
    or
    (
        source_id == "25CS102"
        and target_code == "IT23231"
    )
    or
    (
        source_id == "25ITC01"
        and target_code == "IT23231"
    )

    # NEW: Applied Statistics vs Probability
    or
    (
        source_id == "25CY201"
        and target_code == "MA23214"
    )

    # NEW: Applied Statistics vs Python Lab
    or
    (
        source_id == "25CY201"
        and target_code == "CS23221"
    )

    # NEW: Business Communication vs Technical Communication
    or
    (
        source_id == "25ENC01"
        and target_code == "HS23111"
    )

    # NEW: Mathematical Foundation comparison
    or
    (
        source_id == "25MA102"
        and target_code == "MA23116"
    )

    # NEW: Materials Science vs Physics
    or
    (
        source_id == "25MS101"
        and target_code == "PH23132"
    )
)

            if debug_pair:

                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    "MATCH DEBUG"
                )

                print(
                    f"SOURCE: "
                    f"{source_id} - "
                    f"{source_title}"
                )

                print(
                    f"TARGET: "
                    f"{target_code} - "
                    f"{target_title}"
                )

                print(
                    "TITLE SCORE:",
                    round(
                        title_score,
                        4,
                    ),
                )

                print(
                    f"COURSE GATE: "
                    f"{allowed} "
                    f"({gate_reason})"
                )

                print(
                    f"PAIR COVERAGE "
                    f"@ {topic_threshold}: "
                    f"{pair_coverage}%"
                )

                for result in pair_match_results:

                    print(
                        "\nTARGET TOPIC:"
                    )

                    print(
                        result.get(
                            "target_text"
                        )
                    )

                    print(
                        "\nBEST SOURCE MATCH:"
                    )

                    print(
                        result.get(
                            "best_match_text"
                        )
                    )

                    print(
                        "\nSIMILARITY:",
                        result.get(
                            "similarity"
                        ),
                    )

                    print(
                        "LEXICAL COVERAGE:",
                        result.get(
                            "lexical_coverage"
                        ),
                    )

                    print(
                        "MATCH METHOD:",
                        result.get(
                            "coverage_method"
                        ),
                    )

                    print(
                        "COVERED:",
                        result.get(
                            "covered"
                        ),
                    )

                print(
                    "=" * 80
                    + "\n"
                )

            if not allowed:
                continue

            # -----------------------------------------------------
            # Balanced allocation score
            #
            # Course identity gets the highest importance.
            # Coverage remains important.
            # Average semantic similarity is a smaller tie-breaker.
            # -----------------------------------------------------

            allocation_score = (
                0.55 * title_score
                + 0.35 * (
                    pair_coverage
                    / 100.0
                )
                + 0.10
                * average_similarity
            )

            claims.append(
                {
                    "course_id":
                        course_id,

                    "source_id":
                        source_id,

                    "coverage":
                        pair_coverage,

                    "covered_count":
                        covered_count,

                    "title_score":
                        title_score,

                    "average_similarity":
                        average_similarity,

                    "direct_identity":
                        (
                            title_score
                            >= DIRECT_TITLE_LOCK_THRESHOLD
                        ),

                    "allocation_score":
                        allocation_score,
                }
            )

    # -------------------------------------------------------------
    # Safer source-course allocation
    #
    # PASS 1:
    # Protect obvious direct/same-course matches.
    #
    # PASS 2:
    # Allocate remaining source courses by a balanced score.
    #
    # IMPORTANT:
    #     one source -> one target
    #
    # But:
    #     many sources -> one target
    #
    # is allowed.
    # -------------------------------------------------------------

    source_owner = {}

    source_owner_claim = {}

    target_sources = {
        course[
            "course_id"
        ]: []
        for course
        in courses
    }

    # -------------------------------------------------------------
    # PASS 1 - direct identity locks
    # -------------------------------------------------------------

    direct_claims = [
        claim
        for claim
        in claims
        if claim[
            "direct_identity"
        ]
    ]

    direct_claims.sort(
        key=lambda item: (
            item[
                "title_score"
            ],
            item[
                "coverage"
            ],
            item[
                "average_similarity"
            ],
        ),
        reverse=True,
    )

    for claim in direct_claims:

        source_id = (
            claim[
                "source_id"
            ]
        )

        course_id = (
            claim[
                "course_id"
            ]
        )

        if source_id in source_owner:
            continue

        source_owner[
            source_id
        ] = course_id

        source_owner_claim[
            source_id
        ] = claim

        target_sources[
            course_id
        ].append(
            source_id
        )

    # -------------------------------------------------------------
    # PASS 2 - remaining claims
    # -------------------------------------------------------------

    remaining_claims = [
        claim
        for claim
        in claims
        if claim[
            "source_id"
        ]
        not in source_owner
    ]

    remaining_claims.sort(
        key=lambda item: (
            item[
                "allocation_score"
            ],
            item[
                "coverage"
            ],
            item[
                "title_score"
            ],
            item[
                "average_similarity"
            ],
        ),
        reverse=True,
    )

    for claim in remaining_claims:

        source_id = (
            claim[
                "source_id"
            ]
        )

        course_id = (
            claim[
                "course_id"
            ]
        )

        if source_id in source_owner:
            continue

        source_owner[
            source_id
        ] = course_id

        source_owner_claim[
            source_id
        ] = claim

        target_sources[
            course_id
        ].append(
            source_id
        )

    # -------------------------------------------------------------
    # Allocation debug
    # -------------------------------------------------------------

    print(
        "\n"
        "========== SOURCE ALLOCATION DEBUG =========="
    )

    for (
        source_id,
        target_id,
    ) in source_owner.items():

        source_title = (
            source_metadata[
                source_id
            ][
                "title"
            ]
        )

        target_course = next(
            (
                course
                for course
                in courses
                if course[
                    "course_id"
                ]
                == target_id
            ),
            {},
        )

        claim = (
            source_owner_claim.get(
                source_id,
                {},
            )
        )

        print(
            f"{source_id} - "
            f"{source_title}"
            f"  ->  "
            f"{target_course.get('course_code', target_id)} - "
            f"{target_course.get('course_title', '')}"
        )

        print(
            "   "
            f"title="
            f"{round(claim.get('title_score', 0.0), 3)}, "
            f"coverage="
            f"{claim.get('coverage', 0.0)}%, "
            f"allocation_score="
            f"{round(claim.get('allocation_score', 0.0), 3)}"
        )

    print(
        "=============================================\n"
    )

    # -------------------------------------------------------------
    # Final matching after allocation
    # -------------------------------------------------------------

    final_results = {}

    for course in courses:

        course_id = (
            course[
                "course_id"
            ]
        )

        target_topics = (
            embedded_targets[
                course_id
            ]
        )

        owned_source_ids = (
            target_sources[
                course_id
            ]
        )

        owned_topics = []

        allocated_source_courses = []

        for source_id in owned_source_ids:

            owned_topics.extend(
                embedded_sources[
                    source_id
                ]
            )

            allocated_source_courses.append(
                {
                    "source_course_id":
                        source_id,

                    "source_course_title":
                        source_metadata[
                            source_id
                        ][
                            "title"
                        ],
                }
            )

        (
            raw_match_results,
            _,
        ) = match_course(
            target_topics,
            owned_topics,
            topic_threshold,
        )

        (
            match_results,
            coverage_pct,
        ) = _apply_hybrid_topic_coverage(
            raw_match_results,
            owned_topics,
            topic_threshold,
        )

        decision = (
            decide_with_color(
                coverage_pct,
                accept_threshold,
                partial_threshold,
            )
        )

        reasoning = generate(
            match_results,
            coverage_pct,
            decision[
                "status"
            ],
        )

        final_results[
            course_id
        ] = {
            "match_results":
                match_results,

            "coverage_pct":
                coverage_pct,

            "decision":
                decision,

            "reasoning":
                reasoning,

            "allocated_source_courses":
                allocated_source_courses,
        }

    return final_results


# ---------------------------------------------------------------------
# Public multi-course function
# ---------------------------------------------------------------------

def run_multi_course_evaluation(
    courses: list[dict],
    source_topic_texts: list[str] | None = None,
    topic_threshold: float = 0.75,
    accept_threshold: float = 80.0,
    partial_threshold: float = 40.0,
    source_courses: list[dict] | None = None,
) -> dict:
    """
    Full multi-course evaluation.

    Preferred production call:

        run_multi_course_evaluation(
            courses=target_courses,
            source_courses=completed_source_courses,
            topic_threshold=0.75,
            accept_threshold=80.0,
        )
    """

    if not courses:
        raise ValueError(
            "courses is empty - "
            "nothing to evaluate."
        )

    if source_courses is not None:

        return (
            _run_multi_course_structured(
                courses=courses,
                source_courses=source_courses,
                topic_threshold=topic_threshold,
                accept_threshold=accept_threshold,
                partial_threshold=partial_threshold,
            )
        )

    return (
        _run_multi_course_flat_pool(
            courses=courses,
            source_topic_texts=(
                source_topic_texts
                or []
            ),
            topic_threshold=topic_threshold,
            accept_threshold=accept_threshold,
            partial_threshold=partial_threshold,
        )
    )


# ---------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import json

    print(
        "=" * 70
    )

    print(
        "MOCK SINGLE-COURSE TEST"
    )

    print(
        "=" * 70
    )

    result = (
        run_single_course_evaluation(
            target_topic_texts=[
                (
                    "Introduction to supervised learning, "
                    "linear regression"
                ),
                (
                    "Classification algorithms, logistic "
                    "regression, decision trees"
                ),
                (
                    "Neural networks, backpropagation, "
                    "gradient descent"
                ),
            ],
            source_topic_texts=[
                (
                    "Introduction to regression analysis "
                    "and error minimization"
                ),
                (
                    "Tree-based classifiers and "
                    "categorization problems"
                ),
            ],
            credit_hours_required=4,
            credit_hours_completed=3,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )