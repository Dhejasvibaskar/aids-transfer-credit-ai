# main.py

import os
import shutil
from pathlib import Path
from uuid import uuid4

import psycopg2

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    Depends,
    HTTPException,
)

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from pipeline import run_evaluation, DB_CONFIG

from auth import (
    verify_login,
    create_session,
    require_login,
    active_sessions,
)


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="CredRoute"
)


templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


ALLOWED_EXT = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# HELPER: SAVE UPLOADED FILE
# ============================================================

def save_upload(
    upload: UploadFile,
):
    """
    Save an uploaded file using a generated unique filename.

    Returns the local path of the saved file.
    """

    original_name = (
        upload.filename
        or ""
    )

    ext = Path(
        original_name
    ).suffix.lower()


    if ext not in ALLOWED_EXT:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                "Use PDF, JPG, JPEG or PNG."
            ),
        )


    filename = (
        f"{uuid4().hex}{ext}"
    )


    path = os.path.join(
        UPLOAD_DIR,
        filename,
    )


    with open(
        path,
        "wb",
    ) as file:

        shutil.copyfileobj(
            upload.file,
            file,
        )


    return path


# ============================================================
# HELPER: DELETE TEMPORARY UPLOAD
# ============================================================

def remove_upload(
    path,
):
    """
    Remove temporary uploaded files after processing.
    """

    if (
        path
        and os.path.exists(path)
    ):

        try:
            os.remove(path)

        except OSError:
            pass


# ============================================================
# HELPER: GET ONE EVALUATION BATCH
# ============================================================

def get_evaluation_batch(
    cursor,
    batch_id,
):
    """
    Return the parent evaluation batch.

    One batch represents one complete Upload & Evaluate run.
    Multiple runs for the same student name therefore remain separate.
    """

    cursor.execute(
        """
        SELECT
            id,
            student_name,
            completed_semester,
            created_at
        FROM evaluation_batch
        WHERE id = %s
        """,
        (
            batch_id,
        ),
    )

    row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation batch not found.",
        )

    return {
        "id": row[0],
        "student_name": row[1],
        "completed_semester": row[2],
        "created_at": row[3],
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    """
    Application always starts with the HOD login page.
    """

    return RedirectResponse(
        url="/login",
        status_code=303,
    )


# ============================================================
# LOGIN PAGE
# ============================================================

@app.get(
    "/login",
    response_class=HTMLResponse,
)
async def login_page(
    request: Request,
):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={},
    )


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
):

    if not verify_login(
        username,
        password,
    ):

        return RedirectResponse(
            "/login?error=1",
            status_code=303,
        )


    token = create_session(
        username
    )


    response = RedirectResponse(
        "/evaluations",
        status_code=303,
    )


    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
    )


    return response


# ============================================================
# LOGOUT
# ============================================================

@app.get("/logout")
async def logout(
    request: Request,
):

    token = request.cookies.get(
        "session_token"
    )


    if token:

        active_sessions.pop(
            token,
            None,
        )


    response = RedirectResponse(
        "/login",
        status_code=303,
    )


    response.delete_cookie(
        "session_token"
    )


    return response


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.get(
    "/upload",
    response_class=HTMLResponse,
)
async def upload_page(
    request: Request,
    user=Depends(require_login),
):
    """
    HOD opens this page from the
    '+ New Evaluation' button.

    The existing multi-transcript upload.html
    continues to be used.
    """

    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "user": user,
        },
    )


# ============================================================
# RUN TRANSFER CREDIT EVALUATION
# ============================================================

@app.post(
    "/evaluations/run",
    response_class=HTMLResponse,
)
def evaluate(
    request: Request,

    transcripts: list[UploadFile] = File(...),

    syllabus: UploadFile = File(...),

    completed_semester: int = Form(...),

    percentage: float = Form(...),

    category: str = Form(
        "general"
    ),

    institution_recognition: str = Form(...),

    prior_branch: str = Form(...),

    threshold: float = Form(
        80.0
    ),

    user=Depends(require_login),
):
    """
    Complete evaluation flow:

        transcripts
            ↓
        syllabus
            ↓
        eligibility
            ↓
        OCR/parsing
            ↓
        AI matching
            ↓
        reasoning
            ↓
        PostgreSQL
            ↓
        Student Evaluation page
    """

    # --------------------------------------------------------
    # VALIDATE ACCEPTANCE THRESHOLD
    # --------------------------------------------------------

    if (
        threshold <= 40
        or threshold > 100
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Acceptance threshold must be "
                "greater than 40 and no more "
                "than 100."
            ),
        )


    # --------------------------------------------------------
    # VALIDATE COMPLETED SEMESTER
    # --------------------------------------------------------

    if (
        completed_semester < 1
        or completed_semester > 8
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Completed semester must be "
                "between 1 and 8."
            ),
        )


    # --------------------------------------------------------
    # VALIDATE TRANSCRIPTS
    # --------------------------------------------------------

    if not transcripts:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one transcript "
                "must be uploaded."
            ),
        )


    transcript_paths = []

    syllabus_path = None


    try:

        # ----------------------------------------------------
        # SAVE ALL TRANSCRIPTS
        # ----------------------------------------------------

        for transcript in transcripts:

            transcript_path = (
                save_upload(
                    transcript
                )
            )


            transcript_paths.append(
                transcript_path
            )


        # ----------------------------------------------------
        # SAVE SOURCE SYLLABUS
        # ----------------------------------------------------

        syllabus_path = save_upload(
            syllabus
        )


        # ----------------------------------------------------
        # ELIGIBILITY INFORMATION
        # ----------------------------------------------------

        eligibility_data = {

            "percentage":
                percentage,

            "category":
                category,

            "institution_recognition":
                institution_recognition,

            "prior_branch":
                prior_branch,
        }


        # ----------------------------------------------------
        # RUN AI PIPELINE
        # ----------------------------------------------------

        try:

            result = run_evaluation(

                transcript_paths=
                    transcript_paths,

                syllabus_path=
                    syllabus_path,

                regulation_id=1,

                eligibility_data=
                    eligibility_data,

                accept_threshold=
                    threshold,

                completed_semester=
                    completed_semester,
            )


        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error),
            )


        # ----------------------------------------------------
        # INELIGIBLE STUDENT
        # ----------------------------------------------------

        if not result.get(
            "eligible",
            False,
        ):

            return templates.TemplateResponse(
                request=request,
                name="results.html",
                context={
                    "result":
                        result,

                    "threshold":
                        threshold,

                    "completed_semester":
                        completed_semester,

                    "user":
                        user,
                },
            )


        # ----------------------------------------------------
        # REDIRECT TO THIS EVALUATION BATCH
        # ----------------------------------------------------

        batch_id = result.get(
            "batch_id"
        )

        if batch_id:
            return RedirectResponse(
                url=f"/evaluations/{batch_id}",
                status_code=303,
            )

        # ----------------------------------------------------
        # FALLBACK RESULTS PAGE
        # ----------------------------------------------------

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "result":
                    result,

                "threshold":
                    threshold,

                "completed_semester":
                    completed_semester,

                "user":
                    user,
            },
        )


    finally:

        # ----------------------------------------------------
        # CLEAN TEMPORARY FILES
        # ----------------------------------------------------

        for path in transcript_paths:

            remove_upload(
                path
            )


        remove_upload(
            syllabus_path
        )


# ============================================================
# HOD DASHBOARD - ALL STUDENTS
# ============================================================

@app.get(
    "/evaluations",
    response_class=HTMLResponse,
)
def list_evaluations(
    request: Request,
    user=Depends(require_login),
):
    """
    Show one dashboard row per evaluation batch.

    A repeated upload for the same student name remains a separate
    evaluation because grouping is based on evaluation_batch.id.
    """

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                b.id AS batch_id,
                b.student_name,
                COUNT(e.id) AS course_count,

                CASE
                    WHEN COUNT(*) FILTER (
                        WHERE e.final_decision IS NULL
                    ) = 0
                    THEN 'Reviewed'
                    ELSE 'Pending'
                END AS overall_status,

                b.created_at,
                b.completed_semester

            FROM evaluation_batch b

            JOIN evaluation e
                ON e.batch_id = b.id

            GROUP BY
                b.id,
                b.student_name,
                b.created_at,
                b.completed_semester

            ORDER BY
                b.created_at DESC,
                b.id DESC
            """
        )

        rows = cur.fetchall()
        cur.close()

    finally:
        conn.close()

    evaluations = [
        {
            "id": row[0],
            "student_name": row[1],
            "course_count": row[2],
            "overall_status": row[3],
            "created_at": row[4],
            "completed_semester": row[5],
        }
        for row in rows
    ]

    total_count = len(
        evaluations
    )

    pending_count = sum(
        1
        for evaluation in evaluations
        if evaluation["overall_status"] == "Pending"
    )

    completed_count = sum(
        1
        for evaluation in evaluations
        if evaluation["overall_status"] == "Reviewed"
    )

    # Every listed batch has already completed AI evaluation.
    recommended_count = total_count

    return templates.TemplateResponse(
        request=request,
        name="evaluations.html",
        context={
            "evaluations": evaluations,
            "total_count": total_count,
            "pending_count": pending_count,
            "recommended_count": recommended_count,
            "completed_count": completed_count,
            "user": user,
        },
    )


# ============================================================
# STUDENT EVALUATION - ALL SUBJECTS
# ============================================================

@app.get(
    "/evaluations/{batch_id}",
    response_class=HTMLResponse,
)
def student_evaluation(
    request: Request,
    batch_id: int,
    user=Depends(require_login),
):
    """
    Show only the course evaluations belonging to one upload/evaluation run.
    """

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        batch = get_evaluation_batch(
            cur,
            batch_id,
        )

        cur.execute(
            """
            SELECT
                e.id,
                e.source_course_name,
                c.title,
                e.coverage_pct,
                e.recommended_status,
                e.final_decision,
                c.code,
                c.semester

            FROM evaluation e

            JOIN course c
                ON e.course_id = c.id

            WHERE
                e.batch_id = %s

            ORDER BY
                c.semester NULLS LAST,
                c.code
            """,
            (
                batch_id,
            ),
        )

        rows = cur.fetchall()
        cur.close()

    finally:
        conn.close()

    courses = [
        {
            "id": row[0],
            "source_course_name": (
                row[1]
                or "No source course allocated"
            ),
            "target_course_name": row[2],
            "coverage_pct": row[3],
            "recommended_status": row[4],
            "final_decision": row[5],
            "course_code": row[6],
            "semester": row[7],
        }
        for row in rows
    ]

    student = {
        "id": batch_id,
        "student_name": batch["student_name"],
        "course_count": len(courses),
        "completed_semester": batch["completed_semester"],
        "created_at": batch["created_at"],
        "source_university": "—",
        "program": "—",
        "admission_year": "—",
    }

    return templates.TemplateResponse(
        request=request,
        name="student_evaluation.html",
        context={
            "student": student,
            "courses": courses,
            "user": user,
        },
    )


# ============================================================
# COURSE AI EXPLAINABILITY PAGE
# ============================================================

@app.get(
    "/evaluations/{batch_id}/courses/{evaluation_id}",
    response_class=HTMLResponse,
)
def course_review(
    request: Request,
    batch_id: int,
    evaluation_id: int,
    user=Depends(require_login),
):
    """
    Show the detailed AI explanation for one course within one batch.
    """

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        batch = get_evaluation_batch(
            cur,
            batch_id,
        )

        cur.execute(
            """
            SELECT
                e.id,
                e.student_name,
                e.source_course_name,
                e.source_grade,
                e.credit_hours,
                e.coverage_pct,
                e.recommended_status,
                e.final_decision,
                e.reasoning,
                c.title,
                c.credits,
                c.code,
                c.semester

            FROM evaluation e

            JOIN course c
                ON e.course_id = c.id

            WHERE
                e.id = %s
                AND e.batch_id = %s
            """,
            (
                evaluation_id,
                batch_id,
            ),
        )

        row = cur.fetchone()
        cur.close()

    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Course evaluation not found for this batch.",
        )

    evaluation = {
        "id": row[0],
        "student_id": batch_id,
        "batch_id": batch_id,
        "student_name": row[1],
        "source_course_name": (
            row[2]
            or "No source course allocated"
        ),
        "source_grade": row[3],
        "source_credits": None,
        "coverage_pct": row[5],
        "recommended_status": row[6],
        "final_decision": row[7],
        "reasoning": (
            row[8]
            or "No AI reasoning was stored for this evaluation."
        ),
        "target_course_name": row[9],
        "target_credits": row[10],
        "target_course_code": row[11],
        "semester": row[12],
        "matched_topics": [],
        "missing_topics": [],
    }

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "evaluation": evaluation,
            "user": user,
        },
    )


# ============================================================
# FORM-BASED FINAL DECISION
# ============================================================

@app.post(
    "/evaluations/{eval_id}/decision"
)
def set_decision(
    eval_id: int,
    decision: str = Form(...),
    user=Depends(require_login),
):
    """
    Save a final HOD decision and return to the same evaluation batch.
    """

    decision = decision.strip().lower()

    if decision not in (
        "accept",
        "partial",
        "redo",
        "overridden",
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid final decision.",
        )

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE evaluation
            SET final_decision = %s
            WHERE id = %s
            RETURNING batch_id
            """,
            (
                decision,
                eval_id,
            ),
        )

        updated = cur.fetchone()

        if updated is None:
            raise HTTPException(
                status_code=404,
                detail="Evaluation not found.",
            )

        batch_id = updated[0]

        conn.commit()
        cur.close()

    finally:
        conn.close()

    if batch_id:
        return RedirectResponse(
            url=(
                f"/evaluations/{batch_id}"
                f"/courses/{eval_id}"
            ),
            status_code=303,
        )

    return RedirectResponse(
        "/evaluations",
        status_code=303,
    )


# ============================================================
# JSON DECISION MODEL
# ============================================================

class DecisionUpdate(
    BaseModel
):
    final_decision: str


# ============================================================
# JAVASCRIPT FINAL DECISION
# ============================================================

@app.patch(
    "/evaluations/{eval_id}/decision"
)
def update_decision(
    eval_id: int,
    payload: DecisionUpdate,
    user=Depends(require_login),
):
    """
    Endpoint used by static/js/review.js.

    Supported examples:

        approved
        accept
        partial
        redo
        overridden

    'approved' means:
        accept the AI recommendation as the
        final HOD decision.
    """

    requested_decision = (
        payload.final_decision
        .strip()
        .lower()
    )


    allowed = {
        "approved",
        "accept",
        "partial",
        "redo",
        "overridden",
    }


    if requested_decision not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid final decision."
            ),
        )


    conn = psycopg2.connect(
        **DB_CONFIG
    )


    try:

        cur = conn.cursor()


        # ----------------------------------------------------
        # GET AI RECOMMENDATION
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT
                recommended_status,
                batch_id

            FROM evaluation

            WHERE id = %s
            """,
            (
                eval_id,
            ),
        )


        row = cur.fetchone()


        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation not found."
                ),
            )


        recommended_status = row[0]
        batch_id = row[1]


        # ----------------------------------------------------
        # APPROVE = ACCEPT AI RECOMMENDATION
        # ----------------------------------------------------

        if (
            requested_decision
            == "approved"
        ):

            final_decision = (
                recommended_status
            )


        else:

            final_decision = (
                requested_decision
            )


        # ----------------------------------------------------
        # SAVE FINAL DECISION
        # ----------------------------------------------------

        cur.execute(
            """
            UPDATE evaluation

            SET final_decision = %s

            WHERE id = %s
            """,
            (
                final_decision,
                eval_id,
            ),
        )


        conn.commit()


        cur.close()


    finally:

        conn.close()


    return {
        "success": True,

        "evaluation_id":
            eval_id,

        "final_decision":
            final_decision,

        "ai_recommendation":
            recommended_status,
    }