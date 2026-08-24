"""
In this module, we combine the different scores evaluating different aspects of the answer into a single score.
"""

WEIGHTS = {
    "groundedness": 0.40,
    "completeness": 0.30,
    "correctness": 0.30,
}


def weighted_score(
    groundedness: float,
    completeness: float,
    correctness: float,
) -> float:

    return (
        groundedness * WEIGHTS["groundedness"]
        + completeness * WEIGHTS["completeness"]
        + correctness * WEIGHTS["correctness"]
    )