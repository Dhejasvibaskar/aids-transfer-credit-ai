"""
reasoning_generator.py
------------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Turn matching_engine's raw per-topic results + coverage % + decision
    into a structured, human-readable explanation the HOD can actually
    read — instead of a table of similarity scores.

Design note:
    Not every output field comes from the AI. overlap_pct, missing_units,
    and credit_difference are already known exactly from earlier steps,
    so they're computed directly in Python (zero hallucination risk).
    Gemini is only asked for the genuinely generative parts: the
    narrative, the suggested bridge module, and the learning-outcomes
    judgment — things that actually require language generation/judgment,
    not arithmetic.

Privacy rule (from the build plan):
    Only topic text and scores are ever sent to the Gemini API — no
    student-identifying fields. _sanitize_match_results() enforces this
    even if a caller accidentally passes extra fields in.

Fallback:
    If GEMINI_API_KEY isn't set, or the API call/JSON parse fails, this
    falls back to a simple rule-based narrative instead of crashing —
    so the pipeline never hard-fails just because the LLM had a hiccup.
"""

import os
import json

try:
    from google import genai
except ImportError:
    genai = None


# Only these fields are ever allowed into the Gemini prompt. Anything
# else on a match_results entry (there shouldn't be anything else, but
# just in case) gets dropped here.
ALLOWED_MATCH_KEYS = {"target_text", "best_match_text", "similarity", "covered"}


def _sanitize_match_results(match_results: list[dict]) -> list[dict]:
    """Strip match_results down to only topic text + scores before
    anything is built into a prompt. Defense-in-depth against
    accidentally leaking student-identifying fields."""
    return [
        {k: r.get(k) for k in ALLOWED_MATCH_KEYS if k in r}
        for r in match_results
    ]


def _compute_deterministic_fields(
    match_results: list[dict],
    coverage_pct: float,
    credit_hours_required: float | None,
    credit_hours_completed: float | None,
) -> dict:
    """Fields we already know exactly — computed here, not asked from
    the LLM, so they can never drift from the real numbers."""
    missing_units = [r["target_text"] for r in match_results if not r.get("covered")]

    credit_difference = None
    if credit_hours_required is not None and credit_hours_completed is not None:
        credit_difference = round(credit_hours_required - credit_hours_completed, 2)

    return {
        "overlap_pct": coverage_pct,
        "missing_units": missing_units,
        "credit_difference": credit_difference,
    }


def _build_prompt(
    sanitized_matches: list[dict],
    coverage_pct: float,
    recommended_status: str,
    missing_units: list[str],
) -> str:
    """Build the Gemini prompt. Only topic text/scores go in — no
    student name, ID, or any other identifying field."""
    matches_block = "\n".join(
        f"- Target: \"{m['target_text']}\" | Best match: \"{m.get('best_match_text')}\" "
        f"| Similarity: {m.get('similarity')} | Covered: {m.get('covered')}"
        for m in sanitized_matches
    )
    missing_block = "\n".join(f"- {m}" for m in missing_units) if missing_units else "(none)"

    return f"""You are assisting an academic transfer-credit evaluation. Based on the
topic-matching results below, produce a structured assessment. Respond
with ONLY a valid JSON object — no markdown fences, no preamble, no
explanation outside the JSON.

Topic match results:
{matches_block}

Overall coverage: {coverage_pct}%
Recommended status: {recommended_status}
Missing/uncovered target topics:
{missing_block}

Return exactly this JSON shape:
{{
  "learning_outcomes_match": "full" | "partial" | "none",
  "suggested_bridge_module": "<a short suggested bridge module title covering the gaps, or null if no gap exists>",
  "narrative": "<2-3 sentence plain-language explanation of this recommendation, suitable for a department head to read directly>"
}}

Important rules for edge cases:
- Only suggest a bridge_module if the missing topics are narrow enough
  that ONE additional module could realistically cover them (roughly
  {MAX_MISSING_TOPICS_FOR_BRIDGE} topics or fewer). If more topics than
  that are missing, set suggested_bridge_module to null and say plainly
  in the narrative that the gap is too broad for a single bridge module
  — do not invent an overly broad or vague "bridge module" just to fill
  the field.
- If coverage is very low (near 0%) or no meaningful topics were
  matched at all, do not describe partial learning outcomes that don't
  exist. Be direct that little or nothing was demonstrated to overlap,
  rather than softening the assessment.
- Never state a learning outcome as matched if no corresponding topic
  in the results actually supports it.
"""


def _call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str | None:
    """Call the Gemini API. Returns None (triggering fallback) if the
    SDK isn't installed, no API key is set, or the call fails — rather
    than raising and breaking the pipeline."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if genai is None or not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        print(f"[reasoning_generator] Gemini call failed, using fallback: {e}")
        return None


def _parse_gemini_json(text: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


# A "bridge module" is meant to be ONE additional module that closes the
# gap. If more topics than this are missing, no single module could
# realistically cover that much ground — suggesting one anyway would be
# misleading the HOD into thinking a small top-up fixes a large gap.
MAX_MISSING_TOPICS_FOR_BRIDGE = 2


def _fallback_reasoning(
    coverage_pct: float,
    recommended_status: str,
    missing_units: list[str],
) -> dict:
    """Simple rule-based reasoning used when Gemini is unavailable or
    returns something unparseable. Keeps the pipeline functional today,
    before a real GEMINI_API_KEY is wired up."""
    status_to_outcome = {"accept": "full", "partial": "partial", "redo": "none"}
    learning_outcomes_match = status_to_outcome.get(recommended_status, "none")

    bridge = None
    gap_too_broad_for_bridge = False
    if recommended_status == "partial" and missing_units:
        if len(missing_units) <= MAX_MISSING_TOPICS_FOR_BRIDGE:
            bridge = f"Bridge module covering: {', '.join(missing_units)}"
        else:
            # Edge case: "no bridge module possible" — the gap spans too
            # many topics for one module to plausibly close.
            gap_too_broad_for_bridge = True

    narrative = f"Coverage came out to {coverage_pct}%, giving a recommended status of '{recommended_status}'. "
    if missing_units:
        narrative += f"{len(missing_units)} topic(s) were not sufficiently covered."
        if gap_too_broad_for_bridge:
            narrative += (
                " The gap spans too many topics for a single bridge module — "
                "a full retake or multiple supplementary modules would be needed."
            )
    else:
        narrative += "All required topics were covered."

    return {
        "learning_outcomes_match": learning_outcomes_match,
        "suggested_bridge_module": bridge,
        "narrative": narrative,
    }


def generate(
    match_results: list[dict],
    coverage_pct: float,
    recommended_status: str,
    credit_hours_required: float | None = None,
    credit_hours_completed: float | None = None,
) -> dict:
    """
    Main entry point — matches the shared interface contract.

    Args:
        match_results: per-topic results from matching_engine.match_course().
        coverage_pct: overall coverage % from match_course().
        recommended_status: "accept" | "partial" | "redo" from decision_engine.decide().
        credit_hours_required: target course's credit hours (optional — needed
            for credit_difference to be populated).
        credit_hours_completed: student's completed credit hours for the
            source course(s) (optional, same reason).

    Returns:
        {
          "overlap_pct": float,
          "learning_outcomes_match": "full" | "partial" | "none",
          "missing_units": list[str],
          "credit_difference": float | None,
          "suggested_bridge_module": str | None,
          "narrative": str
        }
    """
    sanitized = _sanitize_match_results(match_results)
    deterministic = _compute_deterministic_fields(
        match_results, coverage_pct, credit_hours_required, credit_hours_completed
    )

    prompt = _build_prompt(
        sanitized, coverage_pct, recommended_status, deterministic["missing_units"]
    )

    ai_fields = None
    raw_response = _call_gemini(prompt)
    if raw_response is not None:
        try:
            ai_fields = _parse_gemini_json(raw_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[reasoning_generator] Failed to parse Gemini JSON, using fallback: {e}")
            ai_fields = None

    if ai_fields is None:
        ai_fields = _fallback_reasoning(
            coverage_pct, recommended_status, deterministic["missing_units"]
        )

    return {
        "overlap_pct": deterministic["overlap_pct"],
        "learning_outcomes_match": ai_fields.get("learning_outcomes_match", "none"),
        "missing_units": deterministic["missing_units"],
        "credit_difference": deterministic["credit_difference"],
        "suggested_bridge_module": ai_fields.get("suggested_bridge_module"),
        "narrative": ai_fields.get("narrative", ""),
    }


# ---------------------------------------------------------------------
# Standalone test — runs fully offline via the fallback path if
# GEMINI_API_KEY isn't set, so this is testable today either way.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    sample_match_results = [
        {"target_text": "Unit 1: Introduction to supervised learning, linear regression",
         "best_match_text": "Introduction to regression analysis and error minimization",
         "similarity": 0.81, "covered": True},
        {"target_text": "Unit 2: Classification algorithms, logistic regression, decision trees",
         "best_match_text": "Tree-based classifiers and their use in categorization problems",
         "similarity": 0.79, "covered": True},
        {"target_text": "Unit 5: Neural networks, backpropagation, gradient descent",
         "best_match_text": None,
         "similarity": 0.31, "covered": False},
    ]
    coverage_pct = 66.67
    recommended_status = "partial"

    if os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY found — calling live Gemini API.\n")
    else:
        print("No GEMINI_API_KEY set — running fallback (rule-based) path.\n")

    result = generate(
        sample_match_results,
        coverage_pct,
        recommended_status,
        credit_hours_required=4,
        credit_hours_completed=3,
    )

    import pprint
    pprint.pprint(result)
