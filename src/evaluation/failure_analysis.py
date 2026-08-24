from collections import Counter


def summarize_failures(results) -> dict:
    """Aggregate failures across evaluation results."""

    type_counts = Counter()
    severity_counts = Counter()

    for result in results:
        for failure in result.failures:
            type_counts[failure.type] += 1
            severity_counts[failure.severity] += 1

    return {
        "by_type": dict(type_counts),
        "by_severity": dict(severity_counts),
        "total": sum(type_counts.values()),
    }