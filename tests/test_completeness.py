from src.evaluation.completeness import (
    get_answerable_obligations,
    completeness_score,
)
from src.evaluation.schemas import ObligationEvaluation


def test_get_answerable_obligations_from_authoritative_evidence():
    obligations = [
        {
            "obligation": "Vacation requires 10 business days notice.",
            "evidence_status": "ANSWERABLE",
            "supporting_claims": [],
            "supporting_evidence": ["LV-003"],
        },
        {
            "obligation": "Managers may approve shorter notice.",
            "evidence_status": "ANSWERABLE",
            "supporting_claims": [],
            "supporting_evidence": ["LV-004"],
        },
    ]

    result = get_answerable_obligations(
        obligations=obligations,
        authoritative_evidence=["LV-003"],
        certified_claims=[],
    )

    assert len(result) == 1
    assert result[0]["obligation"] == (
        "Vacation requires 10 business days notice."
    )


def test_get_answerable_obligations_from_certified_claim():
    obligations = [
        {
            "obligation": "Vacation requires 10 business days notice.",
            "evidence_status": "ANSWERABLE",
            "supporting_claims": [
                "Vacation requests require at least 10 business days of advance notice."
            ],
            "supporting_evidence": [],
        }
    ]

    certified_claims = [
        {
            "claim": (
                "Vacation requests require at least 10 business days "
                "of advance notice."
            ),
            "severity": "critical",
            "supporting_evidence": ["LV-003"],
        }
    ]

    result = get_answerable_obligations(
        obligations=obligations,
        authoritative_evidence=[],
        certified_claims=certified_claims,
    )

    assert len(result) == 1


def test_non_answerable_obligation_is_excluded():
    obligations = [
        {
            "obligation": "Some unavailable information.",
            "evidence_status": "UNANSWERABLE",
            "supporting_claims": [],
            "supporting_evidence": ["LV-999"],
        }
    ]

    result = get_answerable_obligations(
        obligations=obligations,
        authoritative_evidence=["LV-999"],
        certified_claims=[],
    )

    assert result == []


def test_obligation_without_supporting_evidence_or_claim_is_excluded():
    obligations = [
        {
            "obligation": "Unsupported obligation.",
            "evidence_status": "ANSWERABLE",
            "supporting_claims": ["Some claim"],
            "supporting_evidence": ["LV-999"],
        }
    ]

    result = get_answerable_obligations(
        obligations=obligations,
        authoritative_evidence=["LV-003"],
        certified_claims=[],
    )

    assert result == []


def test_completeness_score():
    evaluations = [
        ObligationEvaluation(
            obligation="Obligation 1",
            score=1.0,
        ),
        ObligationEvaluation(
            obligation="Obligation 2",
            score=0.5,
        ),
        ObligationEvaluation(
            obligation="Obligation 3",
            score=0.0,
        ),
    ]

    assert completeness_score(evaluations) == 0.5


def test_completeness_score_all_answered():
    evaluations = [
        ObligationEvaluation("Obligation 1", 1.0),
        ObligationEvaluation("Obligation 2", 1.0),
    ]

    assert completeness_score(evaluations) == 1.0


def test_completeness_score_empty():
    assert completeness_score([]) == 0.0