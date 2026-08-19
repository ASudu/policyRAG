"""Environment-backed configuration for the PolicyRAG application."""

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    generation_model: str = os.getenv("GENERATION_MODEL", "qwen3:8b")
    evaluation_model: str = os.getenv("EVALUATION_MODEL", "gemma3:12b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 120)
    top_k: int = _env_int("TOP_K", 5)
    groundedness_threshold: float = _env_float("GROUNDEDNESS_THRESHOLD", 0.8)
    completeness_threshold: float = _env_float("COMPLETENESS_THRESHOLD", 0.8)
    correctness_threshold: float = _env_float("CORRECTNESS_THRESHOLD", 0.8)
    overall_threshold: float = _env_float("OVERALL_THRESHOLD", 0.8)
    critical_claim_weight: int = _env_int("CRITICAL_CLAIM_WEIGHT", 5)
    high_claim_weight: int = _env_int("HIGH_CLAIM_WEIGHT", 3)
    medium_claim_weight: int = _env_int("MEDIUM_CLAIM_WEIGHT", 2)
    low_claim_weight: int = _env_int("LOW_CLAIM_WEIGHT", 1)


settings = Settings()
