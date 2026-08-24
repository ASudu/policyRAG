"""
In this module, we define utilities to evaluate the completeness. The answer obligations are already extracted in the dataset. We first start with filtering in the answer obligations that are answerable based on the certified retrieved context. Then we check if the answer obligations are covered in the answer. The coverage is determined by checking if the answer obligation is a substring of the answer. We return the number of answer obligations, number of covered answer obligations, and the coverage ratio.
"""
import json

from ollama import chat

from src.config import settings
from src.evaluation.schemas import ObligationEvaluation

_COMPLETENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {"type": "string"},
                    "score": {
                        "type": "number",
                        "enum": [0.0, 0.5, 1.0],
                    },
                    "details": {"type": "string"},
                },
                "required": ["obligation", "score", "details"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}

def get_answerable_obligations(obligations: list[dict], authoritative_evidence: list[str], certified_claims: list[dict],) -> list[dict]:
    """
    Return obligations that can actually be answered from the certified
    evidence and/or certified claims.

    Obligations that are not ANSWERABLE are excluded to avoid false
    completeness failures.
    """

    certified_claim_text = {
        claim["claim"]
        for claim in certified_claims
        if isinstance(claim, dict) and "claim" in claim
    }

    authoritative_evidence_set = set(authoritative_evidence)

    answerable = []

    for obligation in obligations:
        if obligation.get("evidence_status") != "ANSWERABLE":
            continue

        supporting_evidence = set(
            obligation.get("supporting_evidence", [])
        )

        supporting_claims = set(
            obligation.get("supporting_claims", [])
        )

        # Can this obligation be answered from the certified context?
        evidence_available = bool(
            supporting_evidence & authoritative_evidence_set
        )

        # Or from a certified atomic claim?
        claim_available = bool(
            supporting_claims & certified_claim_text
        )

        if evidence_available or claim_available:
            answerable.append(obligation)

    return answerable

def evaluate_obligations(generated_answer: str, answerable_obligations: list[dict], model_name: str | None = None,) -> list[ObligationEvaluation]:
    """
    Evaluate how completely the generated answer satisfies each
    answerable obligation.
    """

    if not generated_answer.strip() or not answerable_obligations:
        return []

    obligations_text = "\n".join(
        f"{i + 1}. {obligation['obligation']}"
        for i, obligation in enumerate(answerable_obligations)
    )

    prompt = f"""
Evaluate whether the generated answer satisfies each answer obligation.

Use exactly one score:

1.0 = ANSWERED
      The answer fully addresses the obligation.

0.5 = PARTIALLY_ANSWERED
      The answer addresses some but not all of the obligation.

0.0 = UNANSWERED
      The answer does not address the obligation.

Judge only whether the answer addresses the obligation.
Do not use outside knowledge.

Answer:
{generated_answer.strip()}

Answer obligations:
{obligations_text}
""".strip()

    response = chat(
        model=model_name or settings.evaluation_model,
        messages=[{"role": "user", "content": prompt}],
        format=_COMPLETENESS_SCHEMA,
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

        obligation = item.get("obligation", "").strip()
        score = item.get("score")

        if not obligation:
            continue

        if score not in {0.0, 0.5, 1.0}:
            continue

        evaluations.append(
            ObligationEvaluation(
                obligation=obligation,
                score=score,
                details=item.get("details", ""),
            )
        )

    return evaluations

def completeness_score(evaluations: list[ObligationEvaluation],) -> float:
    """Average obligation coverage score."""

    if not evaluations:
        return 0.0

    return sum(
        evaluation.score
        for evaluation in evaluations
    ) / len(evaluations)
