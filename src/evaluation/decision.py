# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
PASS, REVIEW, and FAIL governance decision logic.

We define simple thresholds and follow the important rule that:
> A high semantic score cannot override a deterministic policy violation.
"""

PASS_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60


def make_decision(score: float, contradictions: bool) -> str:

    if contradictions:
        return "FAIL"

    if score >= PASS_THRESHOLD:
        return "PASS"

    if score >= REVIEW_THRESHOLD:
        return "REVIEW"

    return "FAIL"