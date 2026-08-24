"""
PASS, REVIEW, and FAIL governance decision logic.

We define simple thresholds and follow the important rule that:
> A high semantic score cannot override a deterministic policy violation.
"""

PASS_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60


def make_decision(score: float, hard_checks_passed: bool) -> str:

    if not hard_checks_passed:
        return "FAIL"

    if score >= PASS_THRESHOLD:
        return "PASS"

    if score >= REVIEW_THRESHOLD:
        return "REVIEW"

    return "FAIL"