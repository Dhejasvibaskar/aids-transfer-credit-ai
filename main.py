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

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
)

from pipeline import (
    run_evaluation,
    DB_CONFIG,
)

from auth import (
    verify_login,
    create_session,
    require_login,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="CredRoute"
)


# ============================================================
# STATIC FILES
# ============================================================

# Ensure the folders exist before mounting them.
os.makedirs(
    "static/images",
    exist_ok=True,
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# UPLOAD SETTINGS
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
# SAVE UPLOAD
# ============================================================

def save_upload(
    upload: UploadFile,
):

    original_name = (
        upload.filename or ""
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
# MAIN EVALUATION PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def upload_page(
    request: Request,
):

    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={},
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
        "session_token",
        token,
        httponly=True,
        samesite="lax",
    )

    return response


# ============================================================
# LOGOUT
# ============================================================

@app.get("/logout")
async def logout():

    response = RedirectResponse(
        "/login",
        status_code=303,
    )

    response.delete_cookie(
        "session_token"
    )

    return response


# ============================================================
# RUN EVALUATION
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
):

    # ========================================================
    # VALIDATE ACCEPTANCE THRESHOLD
    # ========================================================

    if (
        threshold <= 40
        or threshold > 100
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Acceptance threshold must be "
                "greater than 40 and no more than 100."
            ),
        )


    # ========================================================
    # VALIDATE PERCENTAGE
    # ========================================================

    if (
        percentage < 0
        or percentage > 100
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Percentage must be "
                "between 0 and 100."
            ),
        )


    # ========================================================
    # VALIDATE SEMESTER
    # ========================================================

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


    # ========================================================
    # VALIDATE TRANSCRIPTS
    # ========================================================

    if not transcripts:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one transcript "
                "must be uploaded."
            ),
        )


    # ========================================================
    # SAVE TRANSCRIPTS
    # ========================================================

    transcript_paths = []

    for transcript in transcripts:

        transcript_path = save_upload(
            transcript
        )

        transcript_paths.append(
            transcript_path
        )


    # ========================================================
    # SAVE SOURCE SYLLABUS
    # ========================================================

    syllabus_path = save_upload(
        syllabus
    )


    # ========================================================
    # ELIGIBILITY DATA
    # ========================================================

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


    # ========================================================
    # RUN CREDROUTE PIPELINE
    # ========================================================

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


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

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
        },
    )


# ============================================================
# HOD REVIEW QUEUE
# ============================================================

@app.get(
    "/evaluations",
    response_class=HTMLResponse,
)
def list_evaluations(

    request: Request,

    user=Depends(
        require_login
    ),
):

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    try:

        cur = conn.cursor()

        # IMPORTANT:
        # This is your ORIGINAL working DB query.
        # Do not add extra columns until we verify the DB schema.
        cur.execute(
            """
            SELECT
                e.id,
                e.student_name,
                c.title,
                e.coverage_pct,
                e.recommended_status,
                e.final_decision
            FROM evaluation e
            JOIN course c
                ON e.course_id = c.id
            ORDER BY e.created_at DESC
            """
        )

        rows = cur.fetchall()

        cur.close()

    finally:

        conn.close()


    evaluations = [

        {
            "id":
                row[0],

            "student_name":
                row[1],

            "course":
                row[2],

            "coverage_pct":
                row[3],

            "recommended_status":
                row[4],

            "final_decision":
                row[5],
        }

        for row in rows
    ]


    return templates.TemplateResponse(

        request=request,

        name="review.html",

        context={

            "evaluations":
                evaluations,

            "user":
                user,
        },
    )


# ============================================================
# HOD FINAL DECISION
# ============================================================

@app.post(
    "/evaluations/{eval_id}/decision"
)
def set_decision(

    eval_id: int,

    decision: str = Form(...),

    user=Depends(
        require_login
    ),
):

    if decision not in (

        "accept",
        "partial",
        "redo",

    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid decision.",
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
            """,
            (
                decision,
                eval_id,
            ),
        )


        if cur.rowcount == 0:

            conn.rollback()

            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation not found."
                ),
            )


        conn.commit()

        cur.close()

    finally:

        conn.close()


    return RedirectResponse(
        "/evaluations",
        status_code=303,
    )