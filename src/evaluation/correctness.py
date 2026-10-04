# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Semantic correctness evaluation against certified claims."""

import json

from ollama import chat

from src.config import settings
from src.evaluation.prompts import render_prompt
from src.evaluation.schemas import CorrectnessEvaluation


_CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": [
                            "SUPPORTED_BY_CERTIFIED",
                            "CONTRADICTS_CERTIFIED",
                            "NOT_COVERED",
                        ],
                    },
                    "details": {"type": "string"},
                },
                "required": ["claim", "label", "details"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}

_CORRECTNESS_SCORES = {
    "SUPPORTED_BY_CERTIFIED": 1.0,
    "NOT_COVERED": 0.5,
    "CONTRADICTS_CERTIFIED": 0.0,
}



def evaluate_correctness(question: str, certified_claims: list[dict], generated_claims: list[str], model_name: str | None = None,) -> list[CorrectnessEvaluation]:
    """Evaluate generated claims against the certified answer."""

    # No claims => no evaluation
    if not generated_claims or not certified_claims:
        return []

    # Collect the components for evaluation
    certified_claim_text = "\n".join(
        f"- {claim['claim']}"
        for claim in certified_claims
        if isinstance(claim, dict) and claim.get("claim")
    )

    generated_claim_text = "\n".join(
        f"- {claim}"
        for claim in generated_claims
    )

    prompt = render_prompt(
        "correctness",
        QUESTION=question.strip(),
        CERTIFIED_CLAIMS=certified_claim_text,
        GENERATED_CLAIMS=generated_claim_text,
    )

    response = chat(
        model=model_name or settings.evaluation_model,
        messages=[{"role": "user", "content": prompt}],
        format=_CORRECTNESS_SCHEMA,
        options={"temperature": 0},
    )

    content = response.get("message", {}).get("content", "")

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    evaluations = []

    for item in payload.get("evaluations", []):
        if not isinstance(item, dict):
            continue

        claim = item.get("claim", "").strip()
        label = item.get("label", "")
        score = _CORRECTNESS_SCORES.get(label)

        if not claim:
            continue

        if score is None:
            continue

        evaluations.append(
            CorrectnessEvaluation(
                claim=claim,
                label=label,
                score=score,
                details=item.get("details", ""),
            )
        )

    return evaluations


def correctness_score(
    evaluations: list[CorrectnessEvaluation],
) -> float:
    """Calculate correctness score from claim classifications."""

    if not evaluations:
        return 0.0

    return sum(evaluation.score for evaluation in evaluations) / len(evaluations)