"""Atomic factual claim extraction for generated answers."""

import json
from typing import Any

from ollama import chat

from src.config import settings


_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def _normalize_claims(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    seen = set()
    claims: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        claim = item.strip()
        if not claim or claim in seen:
            continue
        seen.add(claim)
        claims.append(claim)
    return claims


def extract_claims(answer: str, model_name: str | None = None) -> list[str]:
    """Extract atomic factual claims from a generated policy answer."""
    if not answer or not answer.strip():
        return []

    prompt = (
        "Extract atomic factual claims from the answer that are truly pertinent to the question/policy context. Exclude any claims that are either general knowledge about the company or irrelevant to the question/policy context. "
        "Return JSON with key 'claims' as an array of short standalone facts. "
        "Exclude opinions, greetings, questions, and paraphrases of the user question.\n\n"
        f"Answer:\n{answer.strip()}"
    )

    response = chat(
        model=model_name or settings.evaluation_model,
        messages=[{"role": "user", "content": prompt}],
        format=_CLAIMS_SCHEMA,
        options={"temperature": 0},
    )

    content = response.get("message", {}).get("content", "")
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    return _normalize_claims(payload.get("claims"))
