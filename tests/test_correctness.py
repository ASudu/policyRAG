# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
from src.evaluation.correctness import correctness_score
from src.evaluation.schemas import CorrectnessEvaluation


def test_correctness_score_all_supported():
    evaluations = [
        CorrectnessEvaluation(
            claim="Vacation requires 10 business days.",
            score=1.0,
            label="SUPPORTED_BY_CERTIFIED",
        ),
        CorrectnessEvaluation(
            claim="Requests must be submitted before vacation.",
            score=1.0,
            label="SUPPORTED_BY_CERTIFIED",
        ),
    ]

    assert correctness_score(evaluations) == 1.0


def test_correctness_score_mixed_results():
    evaluations = [
        CorrectnessEvaluation(
            claim="Claim 1",
            score=1.0,
            label="SUPPORTED_BY_CERTIFIED",
        ),
        CorrectnessEvaluation(
            claim="Claim 2",
            score=0.5,
            label="NOT_COVERED",
        ),
        CorrectnessEvaluation(
            claim="Claim 3",
            score=0.0,
            label="CONTRADICTS_CERTIFIED",
        ),
    ]

    assert correctness_score(evaluations) == 0.5


def test_correctness_score_contradiction():
    evaluations = [
        CorrectnessEvaluation(
            claim="Vacation requires 5 days.",
            score=0.0,
            label="CONTRADICTS_CERTIFIED",
        )
    ]

    assert correctness_score(evaluations) == 0.0


def test_correctness_score_not_covered():
    evaluations = [
        CorrectnessEvaluation(
            claim="Employees receive free lunch.",
            score=0.5,
            label="NOT_COVERED",
        )
    ]

    assert correctness_score(evaluations) == 0.5


def test_correctness_score_empty():
    assert correctness_score([]) == 0.0

def test_correctness_scores_are_valid():
    evaluations = [
        CorrectnessEvaluation(
            claim="Supported claim",
            score=1.0,
            label="SUPPORTED_BY_CERTIFIED",
        ),
        CorrectnessEvaluation(
            claim="Unknown claim",
            score=0.5,
            label="NOT_COVERED",
        ),
        CorrectnessEvaluation(
            claim="Contradictory claim",
            score=0.0,
            label="CONTRADICTS_CERTIFIED",
        ),
    ]

    expected_scores = {
        "SUPPORTED_BY_CERTIFIED": 1.0,
        "NOT_COVERED": 0.5,
        "CONTRADICTS_CERTIFIED": 0.0,
    }

    for evaluation in evaluations:
        assert evaluation.score == expected_scores[evaluation.label]