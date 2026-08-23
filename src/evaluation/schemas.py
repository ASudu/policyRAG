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
class EvaluationResult:
    question: str
    certified_answer: str
    generated_answer: str

    hard_checks: list[HardCheckResult]
    retrieval_metrics: list[RetrievalMetric]

    overall_hard_pass: bool