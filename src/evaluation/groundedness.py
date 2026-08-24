# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
In this module, we implement groundedness evaluation based on retrieved evidence. So, for each generated claim, we check if it is supported by the retrieved evidence. If a claim is not supported by any of the retrieved evidence, it is considered ungrounded.
"""

import json

from ollama import chat

from src.config import settings
from src.evaluation.schemas import ClaimEvaluation, RetrievedEvidence


_GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "score": {
                        "type": "number",
                        "enum": [0.0, 0.5, 1.0],
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "details": {"type": "string"},
                },
                "required": [
                    "claim",
                    "score",
                    "evidence",
                    "details",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}


def evaluate_claim_grounding(claims: list[str], retrieved_evidence: list[RetrievedEvidence], model_name: str | None = None,) -> list[ClaimEvaluation]:
    """Evaluate whether each generated claim is supported by retrieved evidence."""

    # Parameter checks
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise ValueError("claims must be a list of strings")
    if not isinstance(retrieved_evidence, list) or not all(isinstance(e, RetrievedEvidence) for e in retrieved_evidence):
        raise ValueError("retrieved_evidence must be a list of RetrievedEvidence")

    # No claims => no evaluations
    # No retrieved evidence => no evaluations
    if not claims or not retrieved_evidence:
        return []

    # Collect compomnents required for evaluation
    evidence_text = "\n\n".join(f"[{e.section_id}]\n{e.text}" for e in retrieved_evidence)
    claims_text = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))

    prompt = f"""Determine how strongly each claim is supported by the retrieved policy evidence.

Use exactly one score:

1.0 = SUPPORTED (The retrieved evidence fully supports the claim.)

0.5 = PARTIALLY_SUPPORTED (The evidence supports only part of the claim, or the claim contains both supported and unsupported information.)

0.0 = UNSUPPORTED (The retrieved evidence does not support the claim.)

DO NOT use outside knowledge.

For each claim:
- score: 1.0, 0.5, or 0.0
- evidence: section IDs that support the claim
- details: brief explanation

Claims:
{claims_text}

Retrieved evidence:
{evidence_text}
""".strip()

    # LLM call to evaluate claims against retrieved evidence
    response = chat(
        model=model_name or settings.evaluation_model,
        messages=[{"role": "user", "content": prompt}],
        format=_GROUNDING_SCHEMA,
        options={"temperature": 0},
    )

    content = response.get("message", {}).get("content", "")

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    # Process the evaluations from the LLM output
    evaluations = []

    for item in payload.get("evaluations", []):
        if not isinstance(item, dict):
            continue

        claim = item.get("claim", "").strip()

        if not claim:
            continue

        score = item.get("score")

        # Defensively validate the LLM output.
        if score not in {0.0, 0.5, 1.0}:
            continue

        evaluations.append(
            ClaimEvaluation(
                claim=claim,
                score=score,
                evidence=[
                    e for e in item.get("evidence", [])
                    if isinstance(e, str)
                ],
                details=item.get("details", ""),
            )
        )

    return evaluations


def groundedness_score(evaluations: list[ClaimEvaluation],) -> float:
    """Return the average support score across generated claims."""

    if not evaluations:
        return 0.0

    return sum(
        evaluation.score
        for evaluation in evaluations
    ) / len(evaluations)