# pipeline.py

"""
Orchestrates:

eligibility
    ->
OCR/document parsing
    ->
identify completed source courses
    ->
structured source-course embedding/matching/allocation
    ->
decision
    ->
reasoning
    ->
database save
"""

import os
import re

import psycopg2
from dotenv import load_dotenv

from ocr_parser import parse_transcript, parse_document
from eligibility_engine import check_eligibility
from evaluation_pipeline import run_multi_course_evaluation


load_dotenv()


# ---------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------

DB_CONFIG = {
    "dbname": os.getenv(
        "PGDATABASE",
        "credittransfer",
    ),
    "user": os.getenv(
        "PGUSER",
        "postgres",
    ),
    "password": os.getenv(
        "PGPASSWORD",
        "devpass",
    ),
    "host": os.getenv(
        "PGHOST",
        "localhost",
    ),
    "port": int(
        os.getenv(
            "PGPORT",
            "5432",
        )
    ),
}


# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------

DEFAULT_TOPIC_THRESHOLD = 0.75
DEFAULT_ACCEPT_THRESHOLD = 80.0
DEFAULT_PARTIAL_THRESHOLD = 40.0


# ---------------------------------------------------------------------
# REC regulation / target syllabus
# ---------------------------------------------------------------------

def get_regulation_courses(
    regulation_id,
    completed_semester,
):
    """
    Fetch only fixed REC target courses that are applicable
    up to the semester completed by the student.

    Example:
        completed_semester = 2

    Returns only REC Semester 1 + Semester 2 courses.

    Professional/Open elective pool courses have semester=NULL
    and are intentionally excluded from automatic REDO decisions.
    """

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                code,
                title,
                credits,
                semester
            FROM course
            WHERE regulation_id = %s
              AND semester IS NOT NULL
              AND semester <= %s
            ORDER BY semester, code
            """,
            (
                regulation_id,
                completed_semester,
            ),
        )

        courses = cur.fetchall()

        result = []

        for (
            course_id,
            code,
            title,
            credits,
            semester,
        ) in courses:

            cur.execute(
                """
                SELECT text
                FROM topic
                WHERE course_id = %s
                ORDER BY unit_no
                """,
                (
                    course_id,
                ),
            )

            topic_texts = [
                row[0].strip()
                for row in cur.fetchall()
                if row[0]
                and row[0].strip()
            ]

            result.append(
                {
                    "id": course_id,
                    "code": code,
                    "title": title,
                    "credits": credits,
                    "semester": semester,
                    "topic_texts": topic_texts,
                }
            )

        cur.close()

        return result

    finally:
        conn.close()

# ---------------------------------------------------------------------
# Source syllabus helpers
# ---------------------------------------------------------------------

def _get_source_courses(
    syllabus_parsed,
):
    """
    Return source syllabus courses while preserving their structure.

    Expected:

        {
            "courses": [
                {
                    "code": "...",
                    "title": "...",
                    "units": [...]
                }
            ]
        }
    """

    if isinstance(
        syllabus_parsed,
        dict,
    ):
        courses = syllabus_parsed.get(
            "courses",
            [],
        )

    elif isinstance(
        syllabus_parsed,
        list,
    ):
        courses = syllabus_parsed

    else:
        return []

    return [
        course
        for course in courses
        if isinstance(
            course,
            dict,
        )
    ]


def extract_source_topic_texts(
    syllabus_parsed,
):
    """
    Flatten source topics.

    NOTE:
        This function is retained for debugging and compatibility.

        The LIVE multi-course evaluation no longer uses this flattened
        list. It uses structured source courses instead.
    """

    courses = _get_source_courses(
        syllabus_parsed
    )

    texts = []

    for course in courses:

        units = (
            course.get("units")
            or course.get("topics")
            or []
        )

        for unit in units:

            if isinstance(
                unit,
                str,
            ):
                text = unit.strip()

            elif isinstance(
                unit,
                dict,
            ):
                text = str(
                    unit.get("text")
                    or unit.get("title")
                    or unit.get("content")
                    or ""
                ).strip()

            else:
                text = str(
                    unit
                ).strip()

            if (
                text
                and text not in texts
            ):
                texts.append(
                    text
                )

    return texts


# ---------------------------------------------------------------------
# Course comparison helpers
# ---------------------------------------------------------------------

def _normalize_code(
    value,
):
    """
    Normalize course codes for comparison.

    Example:
        25-CS-103 -> 25cs103
    """

    return "".join(
        char.lower()
        for char in str(
            value or ""
        )
        if char.isalnum()
    )


def _normalize_title(
    value,
):
    """
    Normalize course titles for comparison.
    """

    text = str(
        value or ""
    ).lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def _title_similarity(
    title_a,
    title_b,
):
    """
    Small deterministic title similarity helper.

    Used only for linking a transcript course to the same course in
    the uploaded source syllabus.

    It is NOT the semantic transfer-credit matching algorithm.
    """

    a = _normalize_title(
        title_a
    )

    b = _normalize_title(
        title_b
    )

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if (
        a in b
        or b in a
    ):
        return 0.95

    words_a = set(
        a.split()
    )

    words_b = set(
        b.split()
    )

    if not words_a or not words_b:
        return 0.0

    intersection = len(
        words_a
        & words_b
    )

    union = len(
        words_a
        | words_b
    )

    if union == 0:
        return 0.0

    return (
        intersection
        / union
    )


# ---------------------------------------------------------------------
# Transcript lookup
# ---------------------------------------------------------------------

def _find_transcript_course(
    transcript_courses,
    target,
):
    """
    Best-effort transcript row lookup.

    Matching order:

        1. Exact normalized course code
        2. Exact title
        3. Title containment
        4. Token overlap
    """

    target_code = _normalize_code(
        target.get("code")
        or target.get(
            "source_course_id"
        )
    )

    target_title = (
        target.get("title")
        or target.get(
            "source_course_title"
        )
        or ""
    )

    # -------------------------------------------------------------
    # Code match first
    # -------------------------------------------------------------

    if target_code:

        for course in transcript_courses:

            source_code = _normalize_code(
                course.get(
                    "code"
                )
            )

            if (
                source_code
                and source_code
                == target_code
            ):
                return course

    # -------------------------------------------------------------
    # Title match
    # -------------------------------------------------------------

    best_course = None
    best_score = 0.0

    for course in transcript_courses:

        score = _title_similarity(
            target_title,
            course.get(
                "title"
            ),
        )

        if score > best_score:
            best_score = score
            best_course = course

    # Transcript and syllabus are from the same source institution,
    # so a moderately strong title match is acceptable.
    if (
        best_course is not None
        and best_score >= 0.60
    ):
        return best_course

    return None

def merge_transcript_courses(
    transcript_course_lists,
):
    """
    Merge courses extracted from multiple transcript files.

    Duplicate course codes are kept only once.

    If a course code is unavailable, normalized title is used
    as a fallback identity.
    """

    merged = []
    seen = set()

    for courses in transcript_course_lists:

        for course in courses:

            if not isinstance(
                course,
                dict,
            ):
                continue

            code = _normalize_code(
                course.get("code")
            )

            title = _normalize_title(
                course.get("title")
            )

            if code:
                identity = (
                    "code",
                    code,
                )

            elif title:
                identity = (
                    "title",
                    title,
                )

            else:
                continue

            if identity in seen:
                continue

            seen.add(
                identity
            )

            merged.append(
                course
            )

    return merged
# ---------------------------------------------------------------------
# Select only courses the student actually completed
# ---------------------------------------------------------------------

def select_completed_source_courses(
    syllabus_data,
    transcript_courses,
):
    """
    Link source syllabus courses to courses appearing on the transcript.

    IMPORTANT:

        The uploaded syllabus may contain an entire degree curriculum,
        including courses the student has NOT completed.

        Those courses must not be used for transfer-credit evaluation.

    Returns only syllabus courses that can be linked to a transcript
    course.
    """

    syllabus_courses = _get_source_courses(
        syllabus_data
    )

    selected = []
    seen = set()

    for syllabus_course in syllabus_courses:

        transcript_match = (
            _find_transcript_course(
                transcript_courses,
                syllabus_course,
            )
        )

        if transcript_match is None:
            continue

        course_code = (
            syllabus_course.get(
                "code"
            )
            or ""
        )

        course_title = (
            syllabus_course.get(
                "title"
            )
            or ""
        )

        identity = (
            _normalize_code(
                course_code
            ),
            _normalize_title(
                course_title
            ),
        )

        if identity in seen:
            continue

        seen.add(
            identity
        )

        # Keep the original source syllabus structure.
        selected.append(
            syllabus_course
        )

    return selected


# ---------------------------------------------------------------------
# Database save
# ---------------------------------------------------------------------

def save_evaluation(
    student_name,
    course_id,
    source_course_name,
    source_grade,
    source_scale,
    credit_hours,
    coverage_pct,
    recommended_status,
    reasoning_text,
):
    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO evaluation (
                student_name,
                course_id,
                source_course_name,
                source_grade,
                source_scale,
                credit_hours,
                coverage_pct,
                recommended_status,
                reasoning
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s
            )
            RETURNING id
            """,
            (
                student_name,
                course_id,
                source_course_name,
                source_grade,
                source_scale,
                credit_hours,
                coverage_pct,
                recommended_status,
                reasoning_text,
            ),
        )

        eval_id = cur.fetchone()[0]

        conn.commit()

        cur.close()

        return eval_id

    finally:
        conn.close()


# ---------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------

def run_evaluation(
    transcript_paths,
    syllabus_path,
    regulation_id,
    eligibility_data,
    completed_semester,
    accept_threshold=DEFAULT_ACCEPT_THRESHOLD,
    partial_threshold=DEFAULT_PARTIAL_THRESHOLD,
):
    """
    Full student evaluation.

    Supports multiple transcript files and evaluates
    only REC target courses up to completed_semester.
    """

    # -----------------------------------------------------------------
    # 1. Eligibility
    # -----------------------------------------------------------------

    eligibility = check_eligibility(
        eligibility_data
    )

    if not eligibility["eligible"]:
        return {
            "eligible": False,
            "reasons": eligibility["reasons"],
            "student_name": None,
            "completed_semester": completed_semester,
            "courses": [],
            "accepted_courses": [],
            "partial_courses": [],
            "redo_courses": [],
            "summary": {
                "accepted": 0,
                "partial": 0,
                "redo": 0,
                "total": 0,
            },
        }

    # -----------------------------------------------------------------
    # 2. Validate transcript list
    # -----------------------------------------------------------------

    if not transcript_paths:
        raise ValueError(
            "No transcript files were provided."
        )

    # -----------------------------------------------------------------
    # 3. Parse ALL uploaded transcripts
    # -----------------------------------------------------------------

    transcript_course_lists = []

    student_name = None

    for transcript_path in transcript_paths:

        transcript_data = parse_transcript(
            transcript_path
        )

        parsed_name = (
            transcript_data.get(
                "student_name"
            )
        )

        # No same-student validation required.
        # We simply use the first available student name.
        if (
            not student_name
            and parsed_name
        ):
            student_name = parsed_name

        courses = transcript_data.get(
            "courses",
            [],
        )

        if courses:
            transcript_course_lists.append(
                courses
            )

    # -----------------------------------------------------------------
    # 4. Merge transcript courses
    # -----------------------------------------------------------------

    transcript_courses = merge_transcript_courses(
        transcript_course_lists
    )

    student_name = (
        student_name
        or "Unknown"
    )

    if not transcript_courses:
        raise ValueError(
            "The transcript files were parsed, but no completed "
            "courses were extracted."
        )

    # -----------------------------------------------------------------
    # 5. Parse source syllabus
    # -----------------------------------------------------------------

    if syllabus_path:

        syllabus_data = parse_document(
            syllabus_path,
            doc_type="syllabus",
        )

    else:

        syllabus_data = {
            "courses": []
        }

    all_source_courses = _get_source_courses(
        syllabus_data
    )

    if (
        syllabus_path
        and not all_source_courses
    ):
        raise ValueError(
            "The syllabus was parsed, but no usable "
            "source courses were extracted."
        )

    # -----------------------------------------------------------------
    # 6. Keep only completed source courses
    # -----------------------------------------------------------------

    completed_source_courses = (
        select_completed_source_courses(
            syllabus_data,
            transcript_courses,
        )
    )

    if (
        syllabus_path
        and not completed_source_courses
    ):
        raise ValueError(
            "Source syllabus courses were extracted, "
            "but none could be linked to courses on "
            "the student's transcript."
        )

    # -----------------------------------------------------------------
    # Debug output
    # -----------------------------------------------------------------

    print(
        "\n"
        "========== SOURCE COURSE DEBUG =========="
    )

    print(
        f"\nStudent: {student_name}"
    )

    print(
        "\nTranscript files processed:"
    )

    print(
        len(transcript_paths)
    )

    print(
        "\nMerged transcript courses found:"
    )

    print(
        len(transcript_courses)
    )

    for index, course in enumerate(
        transcript_courses,
        start=1,
    ):

        print(
            f"{index}. "
            f"{course.get('code')} - "
            f"{course.get('title')} - "
            f"Grade: {course.get('grade')}"
        )

    print(
        "\nSource syllabus courses parsed:"
    )

    print(
        len(all_source_courses)
    )

    for index, course in enumerate(
        all_source_courses,
        start=1,
    ):

        units = (
            course.get("units")
            or course.get("topics")
            or []
        )

        print(
            f"{index}. "
            f"{course.get('code')} - "
            f"{course.get('title')} "
            f"({len(units)} units)"
        )

    print(
        "\nCompleted source courses selected "
        "for transfer evaluation:"
    )

    print(
        len(completed_source_courses)
    )

    for index, course in enumerate(
        completed_source_courses,
        start=1,
    ):

        units = (
            course.get("units")
            or course.get("topics")
            or []
        )

        print(
            f"{index}. "
            f"{course.get('code')} - "
            f"{course.get('title')} "
            f"({len(units)} units)"
        )

    completed_topic_count = sum(
        len(
            course.get("units")
            or course.get("topics")
            or []
        )
        for course
        in completed_source_courses
    )

    print(
        "\nTopics from completed courses:"
    )

    print(
        completed_topic_count
    )

    print(
        "\nStudent completed through Semester:"
    )

    print(
        completed_semester
    )

    print(
        "=========================================\n"
    )

    # -----------------------------------------------------------------
    # 7. Fetch only applicable REC target courses
    # -----------------------------------------------------------------

    regulation_courses = get_regulation_courses(
        regulation_id,
        completed_semester,
    )

    if not regulation_courses:
        raise ValueError(
            f"No applicable REC courses found for "
            f"regulation_id={regulation_id} through "
            f"Semester {completed_semester}."
        )

    print(
        "\nApplicable REC target courses:"
    )

    print(
        len(regulation_courses)
    )

    for course in regulation_courses:

        print(
            f"Semester {course['semester']} | "
            f"{course['code']} - "
            f"{course['title']}"
        )

    # -----------------------------------------------------------------
    # Target lookup
    # -----------------------------------------------------------------

    course_lookup = {
        course["id"]: course
        for course
        in regulation_courses
    }

    courses_for_eval = [
        {
            "course_id":
                course["id"],

            "course_code":
                course["code"],

            "course_title":
                course["title"],

            "target_topic_texts":
                course["topic_texts"],
        }
        for course
        in regulation_courses
        if course["topic_texts"]
    ]

    if not courses_for_eval:
        raise ValueError(
            "No target syllabus topics are available "
            "for the applicable semesters."
        )

    # -----------------------------------------------------------------
    # 8. Matching
    # -----------------------------------------------------------------

    eval_results = run_multi_course_evaluation(
        courses=courses_for_eval,

        source_courses=(
            completed_source_courses
        ),

        topic_threshold=(
            DEFAULT_TOPIC_THRESHOLD
        ),

        accept_threshold=(
            accept_threshold
        ),

        partial_threshold=(
            partial_threshold
        ),
    )

    # -----------------------------------------------------------------
    # 9. Build/save results
    # -----------------------------------------------------------------

    results = []

    for (
        course_id,
        result,
    ) in eval_results.items():

        target = course_lookup[
            course_id
        ]

        coverage_pct = result[
            "coverage_pct"
        ]

        status = (
            result["decision"]["status"]
            .strip()
            .lower()
        )

        reasoning = result[
            "reasoning"
        ]

        allocated_sources = result.get(
            "allocated_source_courses",
            [],
        )

        # -------------------------------------------------------------
        # Identify primary source course
        # -------------------------------------------------------------

        primary_source = (
            allocated_sources[0]
            if allocated_sources
            else None
        )

        if primary_source:

            source_grade_entry = (
                _find_transcript_course(
                    transcript_courses,
                    {
                        "code":
                            primary_source.get(
                                "source_course_id"
                            ),

                        "title":
                            primary_source.get(
                                "source_course_title"
                            ),
                    },
                )
            )

        else:

            source_grade_entry = None

        # -------------------------------------------------------------
        # Human-readable source course names
        # -------------------------------------------------------------

        if allocated_sources:

            source_course_names = []

            for source in allocated_sources:

                title = source.get(
                    "source_course_title"
                )

                code = source.get(
                    "source_course_id"
                )

                if title and code:

                    display_name = (
                        f"{code} - {title}"
                    )

                elif title:

                    display_name = title

                else:

                    display_name = code

                if (
                    display_name
                    and display_name
                    not in source_course_names
                ):
                    source_course_names.append(
                        display_name
                    )

            source_course_name = (
                ", ".join(
                    source_course_names
                )
                if source_course_names
                else None
            )

        else:

            source_course_name = None

        # -------------------------------------------------------------
        # Save evaluation
        # -------------------------------------------------------------

        eval_id = save_evaluation(
            student_name=student_name,

            course_id=course_id,

            source_course_name=(
                source_course_name
            ),

            source_grade=(
                source_grade_entry.get(
                    "grade"
                )
                if source_grade_entry
                else None
            ),

            source_scale="10-point",

            credit_hours=(
                target["credits"]
            ),

            coverage_pct=(
                coverage_pct
            ),

            recommended_status=(
                status
            ),

            reasoning_text=(
                reasoning.get(
                    "narrative",
                    "",
                )
            ),
        )

        # -------------------------------------------------------------
        # Browser result
        # -------------------------------------------------------------

        results.append(
            {
                "evaluation_id":
                    eval_id,

                "course":
                    target["title"],

                "course_code":
                    target["code"],

                "semester":
                    target["semester"],

                "coverage_pct":
                    coverage_pct,

                "recommended_status":
                    status,

                "decision_color":
                    result[
                        "decision"
                    ][
                        "color"
                    ],

                "reasoning":
                    reasoning,

                "allocated_source_courses":
                    allocated_sources,
            }
        )

    # -----------------------------------------------------------------
    # 10. Sort by semester and course code
    # -----------------------------------------------------------------

    results.sort(
        key=lambda item: (
            item.get("semester") or 999,
            item.get("course_code") or "",
        )
    )

    # -----------------------------------------------------------------
    # 11. HOD-friendly result groups
    # -----------------------------------------------------------------

    accepted_courses = [
        course
        for course in results
        if course[
            "recommended_status"
        ] == "accept"
    ]

    partial_courses = [
        course
        for course in results
        if course[
            "recommended_status"
        ] == "partial"
    ]

    redo_courses = [
        course
        for course in results
        if course[
            "recommended_status"
        ] == "redo"
    ]

    summary = {
        "accepted":
            len(accepted_courses),

        "partial":
            len(partial_courses),

        "redo":
            len(redo_courses),

        "total":
            len(results),
    }

    # -----------------------------------------------------------------
    # 12. Final response
    # -----------------------------------------------------------------

    return {
        "eligible":
            True,

        "reasons":
            [],

        "student_name":
            student_name,

        "completed_semester":
            completed_semester,

        "transcript_count":
            len(transcript_paths),

        "courses":
            results,

        "accepted_courses":
            accepted_courses,

        "partial_courses":
            partial_courses,

        "redo_courses":
            redo_courses,

        "summary":
            summary,
    }