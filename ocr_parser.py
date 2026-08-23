"""
ocr_parser.py
-------------

Purpose:
    Extract structured information from:

    1. Student transcript
    2. Source-university syllabus

Extraction strategy:
    PDF:
        First attempt normal text extraction with pdfplumber.
        If enough readable text exists, Gemini structures that text.

        If the PDF is scanned / has no usable text layer,
        pages are converted to images and Gemini Vision is used.

    Image:
        Gemini Vision is used directly.

Important:
    OCR / extraction happens BEFORE embeddings and matching.

    Transcript output:
        {
            "student_name": str | None,
            "courses": [...]
        }

    Syllabus output:
        {
            "courses": [
                {
                    "code": str | None,
                    "title": str,
                    "units": [...]
                }
            ]
        }
"""

import os
import json
import time

import pdfplumber

from pdf2image import convert_from_path
from PIL import Image

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

MIN_PDF_TEXT_LENGTH = 100


# ---------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------

_client = None


def _get_client():
    """
    Create the Gemini client lazily.

    This avoids failing during application import if the API key
    is temporarily missing.
    """

    global _client

    if _client is not None:
        return _client

    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. "
            "Install it before using Gemini extraction."
        )

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    _client = genai.Client(
        api_key=api_key
    )

    return _client


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

TRANSCRIPT_TEXT_PROMPT = """
You are extracting structured academic information from a student
transcript.

Use ONLY the transcript content provided.

Return ONLY valid JSON.
Do not use markdown fences.
Do not include explanations outside the JSON.

Return this structure:

{
  "student_name": "<student name or null>",
  "courses": [
    {
      "code": "<course code or null>",
      "title": "<course title>",
      "grade": "<grade such as A, B+, O, etc. or null>",
      "grade_point": <numeric grade point or null>,
      "credits": <numeric credit value or null>
    }
  ]
}

Rules:

1. Do not invent courses.
2. Do not infer grades that are not visible.
3. Do not treat semester GPA, CGPA, totals, headers, or footers as courses.
4. Preserve course titles accurately.
5. If a field is unavailable, use null.
"""


TRANSCRIPT_VISION_PROMPT = """
Read this student transcript image carefully.

Extract the student's name and completed academic courses.

Return ONLY valid JSON.
Do not use markdown fences.
Do not include commentary.

Return:

{
  "student_name": "<student name or null>",
  "courses": [
    {
      "code": "<course code or null>",
      "title": "<course title>",
      "grade": "<grade or null>",
      "grade_point": <numeric grade point or null>,
      "credits": <numeric credits or null>
    }
  ]
}

Important:

- Do not invent missing information.
- Ignore headers, GPA totals, CGPA values and administrative text.
- Extract only actual academic courses.
"""


SYLLABUS_TEXT_PROMPT = """
You are extracting course syllabus information for academic
transfer-credit evaluation.

Use ONLY the supplied syllabus text.

Return ONLY valid JSON.
Do not use markdown fences.
Do not provide commentary.

Return this structure:

{
  "courses": [
    {
      "code": "<course code or null>",
      "title": "<course title>",
      "units": [
        "<unit/topic text>",
        "<unit/topic text>"
      ]
    }
  ]
}

Rules:

1. Preserve meaningful syllabus/topic content.
2. Each unit should contain the actual academic content taught.
3. Do not invent missing units or topics.
4. Do not include generic document headers or footers as units.
5. Course outcomes may be omitted when clear unit/topic content exists.
6. If no course code is available, use null.
7. Keep topics sufficiently detailed for semantic similarity matching.
"""


SYLLABUS_VISION_PROMPT = """
Read this syllabus image carefully.

Extract the course or courses and their academic unit/topic content.

Return ONLY valid JSON.
Do not use markdown fences.
Do not provide commentary.

Return:

{
  "courses": [
    {
      "code": "<course code or null>",
      "title": "<course title>",
      "units": [
        "<unit/topic text>",
        "<unit/topic text>"
      ]
    }
  ]
}

Important:

- Preserve actual syllabus content.
- Do not invent topics.
- Ignore page numbers, headers and footers.
- Keep meaningful unit/topic descriptions because these will later be
  converted into embeddings for semantic matching.
"""


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def _clean_json_text(text: str) -> str:
    """
    Remove accidental markdown fences before JSON parsing.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    return cleaned


def _parse_json_response(text: str) -> dict:
    """
    Parse Gemini JSON safely.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned = _clean_json_text(text)

    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    return parsed


# ---------------------------------------------------------------------
# Gemini request
# ---------------------------------------------------------------------

def _call_gemini(contents):
    """
    Call Gemini for transcript / syllabus extraction.

    Temporary 503/server errors are retried.

    429 quota errors and 404 model errors are not repeatedly retried.
    """

    client = _get_client()

    model = os.environ.get(
        "GEMINI_OCR_MODEL",
        "gemini-2.5-flash",
    )

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                },
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty OCR response."
                )

            return _parse_json_response(
                response.text
            )

        except Exception as exc:

            error_text = str(exc)

            # ---------------------------------------------------------
            # Quota exhausted
            # ---------------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                raise RuntimeError(
                    "Gemini OCR quota exhausted. "
                    "Please try again after the quota resets."
                ) from exc

            # ---------------------------------------------------------
            # Model unavailable
            # ---------------------------------------------------------

            if (
                "404" in error_text
                or "NOT_FOUND" in error_text
            ):
                raise RuntimeError(
                    f"Gemini OCR model '{model}' is unavailable."
                ) from exc

            # ---------------------------------------------------------
            # Temporary server overload
            # ---------------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                print(
                    "[ocr_parser] "
                    f"Gemini temporarily unavailable "
                    f"(attempt {attempt}/{max_attempts})."
                )

                if attempt < max_attempts:

                    wait_seconds = 2 * attempt

                    print(
                        "[ocr_parser] "
                        f"Retrying in {wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

            # ---------------------------------------------------------
            # Other temporary/network errors
            # ---------------------------------------------------------

            if attempt < max_attempts:

                wait_seconds = 2 * attempt

                print(
                    "[ocr_parser] "
                    f"Gemini extraction attempt "
                    f"{attempt}/{max_attempts} failed: {exc}"
                )

                print(
                    "[ocr_parser] "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            raise RuntimeError(
                "Unable to extract document using Gemini "
                "after multiple attempts."
            ) from exc

    return None


# ---------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------

def extract_pdf_text(path: str) -> str:
    """
    Extract normal text from a digital PDF using pdfplumber.
    """

    text_parts = []

    with pdfplumber.open(path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text_parts.append(
                    page_text
                )

    return "\n".join(
        text_parts
    ).strip()


def _has_usable_pdf_text(text: str) -> bool:
    """
    Decide whether pdfplumber extracted enough text to avoid Vision OCR.
    """

    if not text:
        return False

    cleaned = " ".join(
        text.split()
    )

    return len(cleaned) >= MIN_PDF_TEXT_LENGTH


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def _load_image(path: str):
    """
    Open an image safely for Gemini Vision.
    """

    image = Image.open(path)

    if image.mode not in (
        "RGB",
        "RGBA",
    ):
        image = image.convert("RGB")

    return image


def _pdf_to_images(path: str):
    """
    Convert scanned PDF pages to PIL images.

    Note:
        pdf2image requires Poppler to be installed on Windows.
    """

    return convert_from_path(
        path
    )


# ---------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------

def extract_transcript_from_text(
    text: str,
) -> dict:
    """
    Structure transcript text extracted from a digital PDF.
    """

    prompt = (
        TRANSCRIPT_TEXT_PROMPT
        + "\n\nTRANSCRIPT CONTENT:\n"
        + text
    )

    parsed = _call_gemini(
        prompt
    )

    parsed.setdefault(
        "student_name",
        None,
    )

    parsed.setdefault(
        "courses",
        [],
    )

    return parsed


def extract_transcript_via_vision(
    path: str,
) -> dict:
    """
    Extract a transcript from an image or scanned PDF.
    """

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension == ".pdf":

        images = _pdf_to_images(
            path
        )

        if not images:
            raise RuntimeError(
                "No pages could be extracted from transcript PDF."
            )

        contents = [
            TRANSCRIPT_VISION_PROMPT,
            *images,
        ]

    elif extension in SUPPORTED_IMAGE_EXTENSIONS:

        image = _load_image(
            path
        )

        contents = [
            TRANSCRIPT_VISION_PROMPT,
            image,
        ]

    else:
        raise ValueError(
            f"Unsupported transcript file type: {extension}"
        )

    parsed = _call_gemini(
        contents
    )

    parsed.setdefault(
        "student_name",
        None,
    )

    parsed.setdefault(
        "courses",
        [],
    )

    return parsed


def parse_transcript(
    path: str,
) -> dict:
    """
    Main transcript parser.

    Digital PDF:
        pdfplumber -> Gemini text structuring

    Scanned PDF / image:
        Gemini Vision
    """

    extension = os.path.splitext(
        path
    )[1].lower()

    # -------------------------------------------------------------
    # PDF
    # -------------------------------------------------------------

    if extension == ".pdf":

        text = extract_pdf_text(
            path
        )

        if _has_usable_pdf_text(
            text
        ):

            parsed = extract_transcript_from_text(
                text
            )

            parsed["method"] = (
                "pdfplumber_text"
            )

            return parsed

        parsed = extract_transcript_via_vision(
            path
        )

        parsed["method"] = (
            "gemini_vision"
        )

        return parsed

    # -------------------------------------------------------------
    # Image
    # -------------------------------------------------------------

    if extension in SUPPORTED_IMAGE_EXTENSIONS:

        parsed = extract_transcript_via_vision(
            path
        )

        parsed["method"] = (
            "gemini_vision"
        )

        return parsed

    raise ValueError(
        f"Unsupported transcript file type: {extension}"
    )


# ---------------------------------------------------------------------
# Syllabus extraction
# ---------------------------------------------------------------------

def extract_syllabus_from_text(
    text: str,
) -> dict:
    """
    Structure syllabus text extracted from a digital PDF.
    """

    prompt = (
        SYLLABUS_TEXT_PROMPT
        + "\n\nSYLLABUS CONTENT:\n"
        + text
    )

    parsed = _call_gemini(
        prompt
    )

    parsed.setdefault(
        "courses",
        [],
    )

    return parsed


def extract_syllabus_via_vision(
    path: str,
) -> dict:
    """
    Extract syllabus content from an image or scanned PDF.
    """

    extension = os.path.splitext(
        path
    )[1].lower()

    if extension == ".pdf":

        images = _pdf_to_images(
            path
        )

        if not images:
            raise RuntimeError(
                "No pages could be extracted from syllabus PDF."
            )

        contents = [
            SYLLABUS_VISION_PROMPT,
            *images,
        ]

    elif extension in SUPPORTED_IMAGE_EXTENSIONS:

        image = _load_image(
            path
        )

        contents = [
            SYLLABUS_VISION_PROMPT,
            image,
        ]

    else:
        raise ValueError(
            f"Unsupported syllabus file type: {extension}"
        )

    parsed = _call_gemini(
        contents
    )

    parsed.setdefault(
        "courses",
        [],
    )

    return parsed


# ---------------------------------------------------------------------
# Syllabus cleanup
# ---------------------------------------------------------------------

def _normalize_course(
    course: dict,
) -> dict:
    """
    Normalize one extracted syllabus course.
    """

    code = course.get(
        "code"
    )

    title = course.get(
        "title"
    )

    units = course.get(
        "units",
        course.get(
            "topics",
            [],
        ),
    )

    if units is None:
        units = []

    if isinstance(
        units,
        str,
    ):
        units = [
            units
        ]

    cleaned_units = []

    for unit in units:

        if isinstance(
            unit,
            str,
        ):

            cleaned = unit.strip()

            if cleaned:
                cleaned_units.append(
                    cleaned
                )

        elif isinstance(
            unit,
            dict,
        ):

            # Support Gemini occasionally returning:
            #
            # {
            #   "title": "...",
            #   "topics": "..."
            # }

            pieces = []

            unit_title = unit.get(
                "title"
            )

            unit_text = (
                unit.get("text")
                or unit.get("topics")
                or unit.get("content")
            )

            if unit_title:
                pieces.append(
                    str(unit_title).strip()
                )

            if unit_text:
                pieces.append(
                    str(unit_text).strip()
                )

            combined = ": ".join(
                piece
                for piece in pieces
                if piece
            )

            if combined:
                cleaned_units.append(
                    combined
                )

    return {
        "code": (
            str(code).strip()
            if code
            else None
        ),
        "title": (
            str(title).strip()
            if title
            else ""
        ),
        "units": cleaned_units,
    }


def _normalize_syllabus(
    parsed: dict,
) -> dict:
    """
    Normalize Gemini syllabus output.
    """

    courses = parsed.get(
        "courses",
        [],
    )

    if not isinstance(
        courses,
        list,
    ):
        courses = []

    normalized_courses = []

    for course in courses:

        if not isinstance(
            course,
            dict,
        ):
            continue

        normalized = _normalize_course(
            course
        )

        # Keep courses that actually contain something useful.
        if (
            normalized["title"]
            or normalized["units"]
        ):
            normalized_courses.append(
                normalized
            )

    return {
        "courses": normalized_courses
    }


# ---------------------------------------------------------------------
# Generic document parser
# ---------------------------------------------------------------------

def parse_document(
    path: str,
    doc_type: str,
) -> dict:
    """
    Parse either a transcript or syllabus.

    Example:

        parse_document(
            "student_syllabus.pdf",
            "syllabus"
        )
    """

    normalized_type = (
        doc_type
        or ""
    ).strip().lower()

    if normalized_type == "transcript":

        return parse_transcript(
            path
        )

    if normalized_type != "syllabus":

        raise ValueError(
            "doc_type must be either "
            "'transcript' or 'syllabus'."
        )

    extension = os.path.splitext(
        path
    )[1].lower()

    # -------------------------------------------------------------
    # Digital PDF
    # -------------------------------------------------------------

    if extension == ".pdf":

        text = extract_pdf_text(
            path
        )

        if _has_usable_pdf_text(
            text
        ):

            parsed = extract_syllabus_from_text(
                text
            )

            normalized = _normalize_syllabus(
                parsed
            )

            normalized["method"] = (
                "pdfplumber_text"
            )

            return normalized

        parsed = extract_syllabus_via_vision(
            path
        )

        normalized = _normalize_syllabus(
            parsed
        )

        normalized["method"] = (
            "gemini_vision"
        )

        return normalized

    # -------------------------------------------------------------
    # Image syllabus
    # -------------------------------------------------------------

    if extension in SUPPORTED_IMAGE_EXTENSIONS:

        parsed = extract_syllabus_via_vision(
            path
        )

        normalized = _normalize_syllabus(
            parsed
        )

        normalized["method"] = (
            "gemini_vision"
        )

        return normalized

    raise ValueError(
        f"Unsupported syllabus file type: {extension}"
    )


# ---------------------------------------------------------------------
# Standalone import / sanity test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "ocr_parser.py loaded successfully."
    )

    print(
        "Gemini configured:",
        bool(
            os.environ.get(
                "GEMINI_API_KEY"
            )
        ),
    )

    print(
        "OCR model:",
        os.environ.get(
            "GEMINI_OCR_MODEL",
            "gemini-2.5-flash",
        ),
    )