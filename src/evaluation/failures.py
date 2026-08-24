from dataclasses import dataclass


class FailureType:
    """Controlled failure taxonomy."""

    POLICY_VIOLATION = "POLICY_VIOLATION"
    NUMERICAL_MISMATCH = "NUMERICAL_MISMATCH"
    DATE_MISMATCH = "DATE_MISMATCH"
    POLARITY_MISMATCH = "POLARITY_MISMATCH"

    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    LOW_CONTEXT_RELEVANCE = "LOW_CONTEXT_RELEVANCE"

    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    PARTIALLY_GROUNDED_CLAIM = "PARTIALLY_GROUNDED_CLAIM"

    INCOMPLETE_ANSWER = "INCOMPLETE_ANSWER"
    PARTIALLY_COMPLETE_ANSWER = "PARTIALLY_COMPLETE_ANSWER"

    CONTRADICTION = "CONTRADICTION"
    UNCOVERED_CLAIM = "UNCOVERED_CLAIM"


@dataclass
class Failure:
    type: str
    severity: str
    details: str = ""

def classify_failures(hard_checks, retrieval_metrics, groundedness_score: float, completeness_score: float, correctness_evaluations,) -> list[Failure]:

    failures = []

    # -------------------------
    # Hard checks
    # -------------------------
    for check in hard_checks:
        if check.passed:
            continue

        failure_type = {
            "policy_reference": FailureType.POLICY_VIOLATION,
            "numerical": FailureType.NUMERICAL_MISMATCH,
            "date": FailureType.DATE_MISMATCH,
            "polarity": FailureType.POLARITY_MISMATCH,
        }.get(check.name)

        if failure_type:
            failures.append(
                Failure(
                    type=failure_type,
                    severity="CRITICAL",
                    details=check.details,
                )
            )

    # -------------------------
    # Retrieval
    # -------------------------
    for metric in retrieval_metrics:
        if metric.name == "evidence_recall" and metric.score < 1.0:
            failures.append(
                Failure(
                    type=FailureType.RETRIEVAL_FAILURE,
                    severity="CRITICAL",
                    details=(
                        f"Evidence recall={metric.score:.2f}. "
                        f"{metric.details}"
                    ),
                )
            )

    # -------------------------
    # Groundedness
    # -------------------------
    if groundedness_score < 1.0:
        severity = (
            "CRITICAL"
            if groundedness_score < 0.5
            else "MAJOR"
        )

        failures.append(
            Failure(
                type=FailureType.UNGROUNDED_CLAIM,
                severity=severity,
                details=(
                    f"Groundedness score={groundedness_score:.2f}"
                ),
            )
        )

    # -------------------------
    # Completeness
    # -------------------------
    if completeness_score < 1.0:
        severity = (
            "CRITICAL"
            if completeness_score < 0.5
            else "MAJOR"
        )

        failures.append(
            Failure(
                type=FailureType.INCOMPLETE_ANSWER,
                severity=severity,
                details=(
                    f"Completeness score={completeness_score:.2f}"
                ),
            )
        )

    # -------------------------
    # Correctness
    # -------------------------
    contradictions = [
        evaluation
        for evaluation in correctness_evaluations
        if evaluation.label == "CONTRADICTS_CERTIFIED"
    ]

    uncovered = [
        evaluation
        for evaluation in correctness_evaluations
        if evaluation.label == "NOT_COVERED"
    ]

    if contradictions:
        failures.append(
            Failure(
                type=FailureType.CONTRADICTION,
                severity="CRITICAL",
                details=(
                    f"{len(contradictions)} generated claim(s) "
                    "contradict certified claims."
                ),
            )
        )

    if uncovered:
        failures.append(
            Failure(
                type=FailureType.UNCOVERED_CLAIM,
                severity="MINOR",
                details=(
                    f"{len(uncovered)} generated claim(s) "
                    "are not covered by certified claims."
                ),
            )
        )

    return failures