# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Atomic factual claim extraction for generated answers."""

import json
from typing import Any

from ollama import chat

from src.config import settings
from src.evaluation.prompts import render_prompt


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


def extract_claims(question: str, answer: str, model_name: str | None = None) -> list[str]:
    """Extract atomic factual claims from a generated policy answer."""
    if not answer or not answer.strip():
        return []

    prompt = render_prompt("claims", QUESTION=question.strip(), ANSWER=answer.strip())

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
