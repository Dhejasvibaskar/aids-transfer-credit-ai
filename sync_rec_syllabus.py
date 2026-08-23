"""
sync_rec_syllabus.py

Safely synchronises the corrected REC syllabus with PostgreSQL.

IMPORTANT:
- Does NOT delete/recreate all course rows.
- Preserves existing course IDs.
- Replaces topic rows with corrected syllabus topics.
- Adds missing official courses.
- Merges duplicate course codes such as CR23A33.
- Adds target semester for fixed REC curriculum courses.
- Professional/Open elective pool courses keep semester = NULL.
- Runs in one transaction.
- Supports --dry-run.
"""

import argparse
import json
import psycopg2

from pipeline import DB_CONFIG


REGULATION_ID = 1


# ---------------------------------------------------------------------
# REC R2023 FIXED CURRICULUM SEMESTER MAPPING
# ---------------------------------------------------------------------
#
# IMPORTANT:
# Only courses that have a fixed semester in the REC curriculum
# are mapped here.
#
# Professional Elective and Open Elective option-pool subjects are
# intentionally NOT mapped because a particular elective can be chosen
# in different elective slots.
# ---------------------------------------------------------------------

COURSE_SEMESTER = {
    # -------------------------
    # Semester I
    # -------------------------
    "HS23111": 1,
    "MA23116": 1,
    "GE23117": 1,
    "PH23132": 1,
    "GE23131": 1,
    "EE23133": 1,
    "MC23111": 1,

    # -------------------------
    # Semester II
    # -------------------------
    "MA23214": 2,
    "GE23217": 2,
    "GE23111": 2,
    "IT23231": 2,
    "CS23231": 2,
    "HS23221": 2,
    "HS23222": 2,
    "GE23123": 2,
    "CS23221": 2,

    # -------------------------
    # Semester III
    # -------------------------
    "MA23313": 3,
    "AI23231": 3,
    "CS23331": 3,
    "CS23332": 3,
    "CS23333": 3,
    "MC23112": 3,

    # -------------------------
    # Semester IV
    # -------------------------
    "MA23434": 4,
    "AI23331": 4,
    "AI23431": 4,
    "AD23431": 4,
    "CS23431": 4,
    "CS23432": 4,
    "GE23421": 4,
    "AD23421": 4,   # Internship

    # -------------------------
    # Semester V
    # -------------------------
    "AD23531": 5,
    "AD23532": 5,
    "AI23531": 5,
    "CS23532": 5,
    "GE23521": 5,

    # -------------------------
    # Semester VI
    # -------------------------
    "AD23631": 6,
    "AD23632": 6,
    "AD23633": 6,
    "GE23627": 6,
    "GE23621": 6,

    # -------------------------
    # Semester VII
    # -------------------------
    "AD23731": 7,
    "CS23633": 7,
    "AD23722": 7,   # Project Phase I

    # -------------------------
    # Semester VIII
    # -------------------------
    "AD23821": 8,   # Project Phase II
}


# ---------------------------------------------------------------------
# Trusted metadata for courses that may not already exist in DB
# ---------------------------------------------------------------------

MISSING_COURSE_METADATA = {
    "GE23117": {
        "title": "Heritage of Tamils",
        "credits": 1,
    },

    "GE23217": {
        "title": "Tamils and Technology",
        "credits": 1,
    },

    "GE23123": {
        "title": (
            "Engineering Practices - Electrical, "
            "Electronics and Computer Systems"
        ),
        "credits": 2,
    },

    "MA23313": {
        "title": "Discrete Mathematics for AI",
        "credits": 4,
    },

    "GE23421": {
        "title": "Soft Skills I",
        "credits": 1,
    },

    "GE23521": {
        "title": "Soft Skills II",
        "credits": 1,
    },

    "AD23A34": {
        "title": "Predictive Analytics",
        "credits": 3,
    },
}


BAD_TEXT_MARKERS = (
    "curriculum and syllabus",
    "course outcomes:",
    "total contact hours",
)


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def load_syllabus(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_syllabus(syllabus):
    if not syllabus:
        raise ValueError("Syllabus JSON is empty.")

    seen_codes = set()

    for course in syllabus:
        code = str(
            course.get("course_code", "")
        ).strip().upper()

        if not code:
            raise ValueError(
                "A course is missing course_code."
            )

        if code in seen_codes:
            raise ValueError(
                f"Duplicate course code inside corrected JSON: {code}"
            )

        seen_codes.add(code)

        topics = course.get("topics")

        if not topics:
            raise ValueError(
                f"{code} has no topics."
            )

        seen_units = set()

        for topic in topics:
            unit_no = topic.get("unit_no")
            text = str(
                topic.get("text", "")
            ).strip()

            if unit_no is None:
                raise ValueError(
                    f"{code} has a topic without unit_no."
                )

            if unit_no in seen_units:
                raise ValueError(
                    f"{code} has duplicate unit_no={unit_no}."
                )

            seen_units.add(unit_no)

            if not text:
                raise ValueError(
                    f"{code} unit {unit_no} has empty text."
                )

            lower = text.lower()

            for marker in BAD_TEXT_MARKERS:
                if marker in lower:
                    raise ValueError(
                        f"{code} unit {unit_no} still contains "
                        f"unwanted text marker: {marker}"
                    )

    print(
        f"Validation OK: {len(syllabus)} unique courses found."
    )


# ---------------------------------------------------------------------
# Database preparation
# ---------------------------------------------------------------------

def ensure_semester_column(cur):
    """
    Safely add semester column if it does not already exist.
    """

    cur.execute(
        """
        ALTER TABLE course
        ADD COLUMN IF NOT EXISTS semester INTEGER
        """
    )


# ---------------------------------------------------------------------
# Course lookup
# ---------------------------------------------------------------------

def find_course_ids(cur, code):
    cur.execute(
        """
        SELECT id
        FROM course
        WHERE regulation_id = %s
          AND UPPER(TRIM(code)) = %s
        ORDER BY id
        """,
        (
            REGULATION_ID,
            code,
        ),
    )

    return [
        row[0]
        for row in cur.fetchall()
    ]


def evaluation_count(cur, course_id):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM evaluation
        WHERE course_id = %s
        """,
        (course_id,),
    )

    return cur.fetchone()[0]


# ---------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------

def merge_duplicate_courses(
    cur,
    code,
    ids,
):
    """
    If the same course code exists more than once,
    keep one canonical row.

    Preference:
      1. course row referenced by most evaluations
      2. lowest ID when tied
    """

    if len(ids) <= 1:
        return ids[0] if ids else None

    counts = []

    for course_id in ids:
        count = evaluation_count(
            cur,
            course_id,
        )

        counts.append(
            (
                course_id,
                count,
            )
        )

    counts.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    keep_id = counts[0][0]

    print(
        f"Duplicate {code}: IDs={ids}. "
        f"Keeping course_id={keep_id}"
    )

    for duplicate_id in ids:

        if duplicate_id == keep_id:
            continue

        # Preserve historical evaluations.
        cur.execute(
            """
            UPDATE evaluation
            SET course_id = %s
            WHERE course_id = %s
            """,
            (
                keep_id,
                duplicate_id,
            ),
        )

        # Remove duplicate topics.
        cur.execute(
            """
            DELETE FROM topic
            WHERE course_id = %s
            """,
            (duplicate_id,),
        )

        # Remove duplicate course row.
        cur.execute(
            """
            DELETE FROM course
            WHERE id = %s
            """,
            (duplicate_id,),
        )

    return keep_id


# ---------------------------------------------------------------------
# Get/create course
# ---------------------------------------------------------------------

def get_or_create_course(
    cur,
    code,
):
    ids = find_course_ids(
        cur,
        code,
    )

    if ids:
        return merge_duplicate_courses(
            cur,
            code,
            ids,
        )

    metadata = MISSING_COURSE_METADATA.get(
        code
    )

    if metadata is None:
        raise RuntimeError(
            f"{code} is missing from PostgreSQL and "
            "no trusted metadata is available for insertion."
        )

    semester = COURSE_SEMESTER.get(code)

    cur.execute(
        """
        INSERT INTO course (
            regulation_id,
            code,
            title,
            credits,
            semester
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            REGULATION_ID,
            code,
            metadata["title"],
            metadata["credits"],
            semester,
        ),
    )

    course_id = cur.fetchone()[0]

    print(
        f"Added missing course: "
        f"{code} -> course_id={course_id}, "
        f"semester={semester}"
    )

    return course_id


# ---------------------------------------------------------------------
# Semester assignment
# ---------------------------------------------------------------------

def update_course_semester(
    cur,
    course_id,
    code,
):
    """
    Assign semester for fixed curriculum courses.

    Elective option-pool subjects are deliberately stored
    as semester NULL.
    """

    semester = COURSE_SEMESTER.get(code)

    cur.execute(
        """
        UPDATE course
        SET semester = %s
        WHERE id = %s
        """,
        (
            semester,
            course_id,
        ),
    )

    return semester


# ---------------------------------------------------------------------
# Main synchronization
# ---------------------------------------------------------------------

def sync_syllabus(
    path,
    dry_run=False,
):
    syllabus = load_syllabus(path)

    validate_syllabus(syllabus)

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        # Safely creates column when needed.
        ensure_semester_column(cur)

        updated_courses = 0
        inserted_topics = 0
        fixed_semester_courses = 0
        elective_or_unmapped_courses = 0

        for course in syllabus:

            code = (
                course["course_code"]
                .strip()
                .upper()
            )

            course_id = get_or_create_course(
                cur,
                code,
            )

            # -------------------------------------------------
            # Set REC semester
            # -------------------------------------------------

            semester = update_course_semester(
                cur,
                course_id,
                code,
            )

            if semester is None:
                elective_or_unmapped_courses += 1
            else:
                fixed_semester_courses += 1

            # -------------------------------------------------
            # Replace OLD topics for only this course
            # -------------------------------------------------

            cur.execute(
                """
                DELETE FROM topic
                WHERE course_id = %s
                """,
                (course_id,),
            )

            # -------------------------------------------------
            # Insert corrected topics
            # -------------------------------------------------

            for topic in course["topics"]:

                cur.execute(
                    """
                    INSERT INTO topic (
                        course_id,
                        unit_no,
                        text
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        course_id,
                        topic["unit_no"],
                        topic["text"].strip(),
                    ),
                )

                inserted_topics += 1

            updated_courses += 1

            semester_display = (
                semester
                if semester is not None
                else "ELECTIVE/UNMAPPED"
            )

            print(
                f"{code}: "
                f"course_id={course_id}, "
                f"semester={semester_display}, "
                f"topics={len(course['topics'])}"
            )

        # -----------------------------------------------------
        # Commit / rollback
        # -----------------------------------------------------

        if dry_run:

            conn.rollback()

            print("\nDRY RUN COMPLETE")
            print(
                "Nothing was written to PostgreSQL."
            )

        else:

            conn.commit()

            print("\nSYNC COMPLETE")
            print(
                "Changes committed to PostgreSQL."
            )

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        print(
            f"Courses processed: "
            f"{updated_courses}"
        )

        print(
            f"Topics inserted: "
            f"{inserted_topics}"
        )

        print(
            f"Fixed-semester courses: "
            f"{fixed_semester_courses}"
        )

        print(
            "Elective/unmapped courses "
            f"(semester=NULL): "
            f"{elective_or_unmapped_courses}"
        )

        cur.close()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "rec_syllabus_export_full_canonical.json"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    sync_syllabus(
        args.input,
        dry_run=args.dry_run,
    )