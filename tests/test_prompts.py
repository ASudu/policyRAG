# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
import pytest

from src.evaluation.prompts import render_prompt


def test_render_prompt_replaces_dynamic_labels():
    prompt = render_prompt(
        "correctness",
        QUESTION="What is the policy?",
        CERTIFIED_CLAIMS="certified claim",
        GENERATED_CLAIMS="generated claim",
    )

    assert "[CERTIFIED_CLAIMS]" not in prompt
    assert "[GENERATED_CLAIMS]" not in prompt
    assert "certified claim" in prompt
    assert "generated claim" in prompt


def test_render_prompt_requires_all_dynamic_labels():
    with pytest.raises(ValueError, match="CERTIFIED_CLAIMS"):
        render_prompt("correctness", GENERATED_CLAIMS="generated claim")


def test_render_prompt_rejects_unknown_prompt():
    with pytest.raises(KeyError, match="Unknown evaluation prompt"):
        render_prompt("unknown")