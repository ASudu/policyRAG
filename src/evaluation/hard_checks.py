# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Deterministic checks for policy section references."""

import re

from src.evaluation.schemas import HardCheckResult


# ================================================================
# Hard checks for policy section references
# ================================================================

POLICY_REFERENCE_PATTERN = re.compile(r"\b[A-Z]{2,3}-\d{3}\b")

def extract_policy_references(text: str) -> set[str]:
    """Return unique policy section references found in answer text."""
    return set(POLICY_REFERENCE_PATTERN.findall(text))


def check_policy_references(
    certified_answer: str,
    generated_answer: str,
) -> HardCheckResult:
    """Check that the generated answer includes every certified reference."""
    expected = extract_policy_references(certified_answer)
    actual = extract_policy_references(generated_answer)
    missing = expected - actual

    return HardCheckResult(
        name="policy_reference",
        passed=not missing,
        details=(
            f"Expected={sorted(expected)}, "
            f"Actual={sorted(actual)}, "
            f"Missing={sorted(missing)}"
        ),
    )

# ================================================================
# Hard checks for numerical values in policy sections
# ================================================================

NUMBER_PATTERN = re.compile(
    r"(?P<currency>\$)?(?P<value>\d+(?:\.\d+)?)"
    r"(?:[- ]+(?P<unit>business[- ]days?|calendar[- ]days?|"
    r"days?|hours?|months?|years?|characters?))?\b",
    re.IGNORECASE,
)

UNIT_NORMALIZATION = {
    "business day": "business days",
    "business days": "business days",
    "calendar day": "calendar days",
    "calendar days": "calendar days",
    "day": "days",
    "days": "days",
    "hour": "hours",
    "hours": "hours",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
    "character": "characters",
    "characters": "characters",
}

def extract_numbers(text: str) -> set[str]:
    """Return normalized numerical values with their units."""
    values = set()
    for match in NUMBER_PATTERN.finditer(text):
        unit = match.group("unit")
        if unit:
            normalized_unit = UNIT_NORMALIZATION[unit.lower().replace("-", " ")]
            values.add(f"{match.group('value')} {normalized_unit}")
        elif match.group("currency"):
            values.add(f"{match.group('value')} USD")
    return values


def check_numbers(
    certified_answer: str,
    generated_answer: str,
) -> HardCheckResult:

    expected = extract_numbers(certified_answer)
    actual = extract_numbers(generated_answer)

    missing = expected - actual

    return HardCheckResult(
        name="numerical",
        passed=not missing,
        details=(
            f"Expected={sorted(expected)}, "
            f"Actual={sorted(actual)}, "
            f"Missing={sorted(missing)}"
        ),
    )

# ================================================================
# Hard checks for dates in policy sections
# ================================================================

DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

def extract_dates(text: str) -> set[str]:
    return set(DATE_PATTERN.findall(text))

def check_dates(
    certified_answer: str,
    generated_answer: str,
) -> HardCheckResult:

    expected = extract_dates(certified_answer)
    actual = extract_dates(generated_answer)

    missing = expected - actual

    return HardCheckResult(
        name="date",
        passed=not missing,
        details=f"Missing dates: {sorted(missing)}",
    )

# ================================================================
# Hard checks for polarity
# ================================================================

POLARITY_PATTERNS = (
    ("prohibition", re.compile(
        r"\b(?:may not|must not|cannot|can't|not permitted|not allowed|"
        r"prohibited|forbidden)\b", re.IGNORECASE,
    )),
    ("non_requirement", re.compile(
        r"\b(?:not required|need not|does not have to|do not have to)\b",
        re.IGNORECASE,
    )),
    ("requirement", re.compile(
        r"\b(?:required|must|need to|has to|have to)\b", re.IGNORECASE,
    )),
    ("permission", re.compile(
        r"\b(?:may|permitted|allowed|can)\b", re.IGNORECASE,
    )),
)

POLARITY_STOP_WORDS = {
    "a", "an", "and", "are", "be", "by", "for", "in", "is", "it",
    "of", "on", "or", "the", "to", "when", "with",
}


def _polarity_statements(text: str) -> list[tuple[str, set[str]]]:
    statements = []
    for sentence in re.split(r"[.!?;]\s*", text):
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        context = words - POLARITY_STOP_WORDS
        for polarity, pattern in POLARITY_PATTERNS:
            if pattern.search(sentence):
                statements.append((polarity, context))
                break
    return statements


def check_polarity(
    certified_answer: str,
    generated_answer: str,
) -> HardCheckResult:
    """Detect polarity reversals in statements sharing meaningful context."""
    expected = _polarity_statements(certified_answer)
    actual = _polarity_statements(generated_answer)
    reversals = []

    for expected_polarity, expected_context in expected:
        matches = [
            actual_polarity
            for actual_polarity, actual_context in actual
            if len(expected_context & actual_context) >= 2
        ]
        if matches and all(actual_polarity != expected_polarity
                           for actual_polarity in matches):
            reversals.append((expected_polarity, sorted(set(matches))))

    return HardCheckResult(
        name="polarity",
        passed=not reversals,
        details=f"Expected={expected}, Actual={actual}, Reversals={reversals}",
    )

# ================================================================
# MAIN FUNCTION TO RUN ALL HARD CHECKS
# ================================================================
def evaluate_hard_checks(
    certified_answer,
    generated_answer,
):
    return [
        check_policy_references(certified_answer, generated_answer),
        check_numbers(certified_answer, generated_answer),
        check_dates(certified_answer, generated_answer),
        check_polarity(certified_answer, generated_answer),
    ]