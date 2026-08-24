# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
Retrieval evidence recall and context relevance evaluation.
"""

from src.evaluation.schemas import RetrievedEvidence, RetrievalMetric

# ================================================
#  Evidence recall
# ================================================

def evidence_recall(
    authoritative_evidence: list[str],
    retrieved_evidence: list[RetrievedEvidence],
) -> RetrievalMetric:
    """
    Measure how much of the authoritative evidence was retrieved.

    authoritative_evidence contains section IDs from the SME-reviewed dataset.
    RetrievedEvidence.section_id is used as the canonical evidence identifier.
    """

    expected = set(authoritative_evidence)
    retrieved = {
        evidence.section_id
        for evidence in retrieved_evidence
    }

    if not expected:
        return RetrievalMetric(
            name="evidence_recall",
            score=0.0,
            details="No authoritative evidence provided.",
        )

    matched = expected & retrieved
    score = len(matched) / len(expected)

    missing = expected - retrieved

    return RetrievalMetric(
        name="evidence_recall",
        score=score,
        details=(
            f"Expected={sorted(expected)}, "
            f"Retrieved={sorted(retrieved)}, "
            f"Matched={sorted(matched)}, "
            f"Missing={sorted(missing)}"
        ),
    )

# ================================================
#  Context relevance
# ================================================

def context_relevance(
    retrieved_evidence: list[RetrievedEvidence],
) -> RetrievalMetric:
    """
    Estimate context relevance from Chroma cosine distances.

    Assumes Chroma is configured to return cosine distance,
    where similarity ~= 1 - distance.
    """

    if not retrieved_evidence:
        return RetrievalMetric(
            name="context_relevance",
            score=0.0,
            details="No retrieved evidence.",
        )

    similarities = [
        1.0 - evidence.distance
        for evidence in retrieved_evidence
    ]

    score = sum(similarities) / len(similarities)

    return RetrievalMetric(
        name="context_relevance",
        score=score,
        details=(
            f"Retrieved={len(retrieved_evidence)}, "
            f"Mean cosine similarity={score:.4f}"
        ),
    )


# ================================================
#  MAIN EVALUATION FUNCTION
# ================================================

def evaluate_retrieval(
    authoritative_evidence: list[str],
    retrieved_evidence: list[RetrievedEvidence],
) -> list[RetrievalMetric]:

    return [
        evidence_recall(
            authoritative_evidence,
            retrieved_evidence,
        ),
        context_relevance(
            retrieved_evidence,
        ),
    ]