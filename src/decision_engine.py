"""
decision_engine.py
--------------------
Owner: Person B (AI / Matching Engine)

Purpose:
    Turn a coverage % (from matching_engine.match_course()) into one of
    three recommended outcomes: "accept", "partial", or "redo".

    Two adjustable thresholds control the boundaries:
      - accept_threshold  (default 80%): coverage at/above this -> accept
      - partial_threshold (default 40%): coverage at/above this (but
        below accept_threshold) -> partial; below this -> redo

    These are function parameters, not hardcoded constants, so the HOD
    can tune them per-department or per-regulation without code changes.
    This does NOT make the final call — the HOD still confirms via
    PATCH /evaluations/{id}/decision. This only produces a recommendation.
"""

# Sensible defaults per the build plan. Kept as module-level constants so
# they're easy to find/tune in one place, but every call site can still
# override them explicitly.
DEFAULT_ACCEPT_THRESHOLD = 80.0
DEFAULT_PARTIAL_THRESHOLD = 40.0

VALID_STATUSES = ("accept", "partial", "redo")

# Traffic-light color for UI display (e.g. a colored badge next to each
# evaluation row). This is purely a presentation layer on top of the
# status returned by decide() — it doesn't change the underlying logic
# or the value written to evaluation.recommended_status.
STATUS_COLOR_MAP = {
    "accept": "green",
    "partial": "yellow",
    "redo": "red",
}


def status_to_color(status: str) -> str:
    """Map a decision status to its traffic-light color.

    Args:
        status: one of "accept", "partial", "redo".

    Returns:
        "green", "yellow", or "red".

    Raises:
        ValueError: if status isn't one of the three valid values.
    """
    if status not in STATUS_COLOR_MAP:
        raise ValueError(
            f"Unknown status '{status}'. Must be one of {VALID_STATUSES}."
        )
    return STATUS_COLOR_MAP[status]


def decide(
    coverage_pct: float,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> str:
    """
    Map a coverage % to a recommended status.

    Args:
        coverage_pct: 0-100, from matching_engine.match_course().
        accept_threshold: coverage at/above this -> "accept". Default 80.
        partial_threshold: coverage at/above this (but below
            accept_threshold) -> "partial"; below it -> "redo". Default 40.

    Returns:
        One of "accept", "partial", "redo".

    Raises:
        ValueError: if coverage_pct is out of [0, 100], or if the two
            thresholds are ordered incorrectly (partial must be < accept).
    """
    if not (0.0 <= coverage_pct <= 100.0):
        raise ValueError(f"coverage_pct must be between 0 and 100, got {coverage_pct}")

    if not (0 <= accept_threshold <= 100):
        raise ValueError(f"accept_threshold must be between 0 and 100, got {accept_threshold}"
        )
    if not (0 <= partial_threshold <= 100):
        raise ValueError(f"partial_threshold must be between 0 and 100, got {partial_threshold}"
        )
    if partial_threshold >= accept_threshold:
        raise ValueError(
            f"partial_threshold ({partial_threshold}) must be lower than "
            f"accept_threshold ({accept_threshold})."
        )
    

    if coverage_pct >= accept_threshold:
        return "accept"
    elif coverage_pct >= partial_threshold:
        return "partial"
    else:
        return "redo"


def decide_with_color(
    coverage_pct: float,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> dict:
    """
    Convenience wrapper: run decide() and attach the matching color in
    one call. Useful for the UI layer, which likely wants both the
    status text and a badge color without calling two functions.

    Returns:
        {"status": "accept" | "partial" | "redo", "color": "green" | "yellow" | "red"}
    """
    status = decide(coverage_pct, accept_threshold, partial_threshold)
    return {"status": status, "color": status_to_color(status)}


# ---------------------------------------------------------------------
# Standalone sanity check — synthetic cases covering full/partial/no
# match, plus the boundary edges exactly at each threshold.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        (100.0, "accept"),   # full match
        (95.0, "accept"),
        (80.0, "accept"),    # exactly at accept threshold -> inclusive
        (79.9, "partial"),   # just under accept threshold
        (65.0, "partial"),   # partial match
        (40.0, "partial"),   # exactly at partial threshold -> inclusive
        (39.9, "redo"),      # just under partial threshold
        (10.0, "redo"),      # no match
        (0.0, "redo"),       # zero coverage (e.g. empty transcript case)
    ]

    print("Running synthetic decision cases...\n")
    all_passed = True
    for coverage, expected in cases:
        result = decide_with_color(coverage)
        check = "PASS" if result["status"] == expected else "FAIL"
        if result["status"] != expected:
            all_passed = False
        print(f"[{check}] coverage={coverage:>5}% -> {result['status']:<8} "
              f"({result['color']:<6}) (expected {expected})")

    print("\nAll cases passed." if all_passed else "\nSome cases FAILED.")
