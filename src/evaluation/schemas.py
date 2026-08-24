# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
In this module, we define the structure and schemas required for the RAG evaluation pipeline. The schemas are designed to facilitate the organization and validation of data used in the evaluation process, ensuring that the inputs and outputs adhere to the expected formats and types.
"""

from dataclasses import dataclass


@dataclass
class RetrievedEvidence:
    chunk_id: str
    document_id: str
    section_id: str # the chunk_id in the dataset is same as the section_id in the document
    text: str
    distance: float


@dataclass
class GeneratedResponse:
    answer: str
    retrieved_evidence: list[RetrievedEvidence]


@dataclass
class HardCheckResult:
    """Represents the result of a hard check."""
    name: str
    passed: bool
    details: str = ""


@dataclass
class RetrievalMetric:
    name: str
    score: float
    details: str = ""

@dataclass
class ClaimEvaluation:
    claim: str
    score: float  # 1.0 = supported, 0.5 = partially supported, 0.0 = unsupported
    evidence: list[str]
    details: str = ""


@dataclass
class ObligationEvaluation:
    """Evaluation of whether a generated answer satisfies an obligation."""
    obligation: str
    score: float # 1.0 = answered, 0.5 = partially answered, 0.0 = unanswered
    details: str = ""


@dataclass
class CorrectnessEvaluation:
    """Evaluation of a generated claim against certified claims."""
    claim: str
    score: float # 1.0 = supported by certified claims, 0.5 = not covered, 0.0 = contradicts certified claims
    label: str
    details: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for one QA pair."""

    question: str
    certified_answer: str
    generated_answer: str

    # Retrieval evaluation
    retrieval_metrics: list[RetrievalMetric]

    # Answer evaluation
    groundedness: float
    completeness: float
    correctness: float

    # Deterministic checks
    hard_checks: list[HardCheckResult]
    overall_hard_pass: bool

    # Detailed semantic evaluations
    claim_evaluations: list[ClaimEvaluation]
    obligation_evaluations: list[ObligationEvaluation]
    correctness_evaluations: list[CorrectnessEvaluation]

    # Final governance
    weighted_score: float
    failures: list[str]
    decision: str