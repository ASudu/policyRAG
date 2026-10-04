# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Load and render version-controlled evaluation judge prompts."""

import json
import re
from functools import lru_cache
from pathlib import Path


_PROMPTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "evaluation" / "prompts.json"
)
_LABEL_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, str]:
    """Load and validate the external evaluation prompt templates."""
    with _PROMPTS_PATH.open(encoding="utf-8") as prompt_file:
        prompts = json.load(prompt_file)

    if not isinstance(prompts, dict) or not all(
        isinstance(name, str) and isinstance(template, str)
        for name, template in prompts.items()
    ):
        raise ValueError("Evaluation prompts must be a JSON object of strings")

    return prompts


def render_prompt(name: str, **values: str) -> str:
    """Render a named prompt by replacing its bracketed dynamic-data labels."""
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Unknown evaluation prompt: {name}")

    template = prompts[name]
    labels = set(_LABEL_PATTERN.findall(template))
    missing_labels = labels - values.keys()
    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"Missing prompt values for {name}: {missing}")

    rendered = template
    for label, value in values.items():
        rendered = rendered.replace(f"[{label}]", value)

    unresolved_labels = _LABEL_PATTERN.findall(rendered)
    if unresolved_labels:
        unresolved = ", ".join(sorted(set(unresolved_labels)))
        raise ValueError(f"Unresolved prompt labels for {name}: {unresolved}")

    return rendered.strip()