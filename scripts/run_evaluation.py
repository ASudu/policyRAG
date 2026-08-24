# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Run one end-to-end RAG evaluation against a QA pair."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.embeddings import EmbeddingModel
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.vectorstore import ChromaVectorStore

from src.evaluation.eval_orchestrate import evaluate
from src.evaluation.schemas import RetrievedEvidence

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(ROOT / "data" / "chroma"),
)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "llama3.2:3b",
)

QA_PATH = ROOT / "data" / "golden_qa.jsonl"


def main() -> None:

    # =========================================================
    # 1. Initialize RAG pipeline
    # =========================================================

    embedder = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIR
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    generator = Generator(
        model_name=GENERATION_MODEL
    )

    print("RAG pipeline initialized.")

    # =========================================================
    # 2. Load one certified QA pair
    # =========================================================

    qa_data = [
        json.loads(line)
        for line in QA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    qa = qa_data[0]

    question = qa["question"]

    print("\nQuestion:")
    print(question)

    # =========================================================
    # 3. Retrieve
    # =========================================================

    chunks = retriever.retrieve(
        query=question,
        top_k=5,
    )

    retrieved_evidence = [
        RetrievedEvidence(
            chunk_id=str(chunk.chunk_id),
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            text=chunk.text,
            distance=chunk.distance,
        )
        for chunk in chunks
    ]

    print("\nRetrieved chunks:")

    for evidence in retrieved_evidence:
        print(
            f"- {evidence.section_id} "
            f"(distance={evidence.distance:.4f})"
        )

    # =========================================================
    # 4. Generate
    # =========================================================

    generated_answer = generator.generate(
        query=question,
        chunks=chunks,
    )

    print("\nGenerated answer:")
    print(generated_answer)

    # =========================================================
    # 5. Run evaluation
    # =========================================================

    evaluation_result = evaluate(
        qa=qa,
        generated_answer=generated_answer,
        retrieved_evidence=retrieved_evidence,
    )

    # =========================================================
    # 6. Print summary
    # =========================================================

    print("\n" + "=" * 50)
    print("EVALUATION RESULT")
    print("=" * 50)

    print(f"Question:      {question}")
    print(f"Groundedness:  {evaluation_result.groundedness:.3f}")
    print(f"Completeness:  {evaluation_result.completeness:.3f}")
    print(f"Correctness:   {evaluation_result.correctness:.3f}")
    print(f"Overall score: {evaluation_result.weighted_score:.3f}")
    print(f"Decision:      {evaluation_result.decision}")

    print("\nFailures:")

    if not evaluation_result.failures:
        print("- None")
    else:
        for failure in evaluation_result.failures:
            print(
                f"- [{failure.severity}] "
                f"{failure.type}: "
                f"{failure.details}"
            )


if __name__ == "__main__":
    main()