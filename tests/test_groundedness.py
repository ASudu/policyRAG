"""
Test groundedness evaluation functions.
"""
from src.evaluation.groundedness import groundedness_score
from src.evaluation.schemas import ClaimEvaluation


def test_groundedness_empty():
    assert groundedness_score([]) == 0.0

def test_groundedness_score():
    evaluations = [
        ClaimEvaluation(
            claim="Claim 1",
            score=1.0,
            evidence=["LV-003"],
        ),
        ClaimEvaluation(
            claim="Claim 2",
            score=0.5,
            evidence=["LV-004", "LV-005"],
),
        ClaimEvaluation(
            claim="Claim 3",
            score=0.0,
            evidence=[],
        ),
    ]

    assert groundedness_score(evaluations) == 0.5

def main():
    test_groundedness_empty()
    test_groundedness_score()
    print("All tests passed.")

if __name__ == "__main__":
    main()