"""Tests for retrieval."""

import os
from dotenv import load_dotenv
from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import Retriever
from src.rag.vectorstore import ChromaVectorStore

load_dotenv()  # Load environment variables from .env file

ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_MODEL_NAME = str(os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
CHROMA_PERSIST_DIRECTORY = str(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma"))


def main() -> None:
    embedder = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME,
    )

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    query = "What are the rules for remote work?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Distance:  {result.distance:.4f}")
        print(f"Policy:    {result.policy_id}")
        print(f"Section:   {result.section_id}")
        print(f"Source:    {result.source}")
        print(f"Text:      {result.text}\n")


if __name__ == "__main__":
    main()