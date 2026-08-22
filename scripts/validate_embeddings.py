"""Validate embedding generation over the policy corpus."""

import os
import sys
import ollama
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunker import Chunker
from src.rag.embeddings import EmbeddingModel
from src.rag.loader import Loader


DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(ROOT / "data" / "documents")))

EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_SECTION_COUNT = 92

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "180"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Probe the embedding model to determine the expected dimension of the embeddings.
response = ollama.embed(
            model=EMBEDDING_MODEL,
            input="dimension probe",
        )
EXPECTED_EMBEDDING_DIMENSION = len(response["embeddings"][0])


def main() -> int:
    loader = Loader()
    chunker = Chunker(
        max_chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL)

    errors = []

    # Load and chunk the corpus.
    policy_paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    documents = loader.load_corpus(DOCUMENTS_DIR)
    chunks = [
        chunk
        for document in documents
        for chunk in chunker.chunk_text(document)
    ]

    # Basic corpus checks.
    if len(policy_paths) != EXPECTED_DOCUMENT_COUNT:
        errors.append(
            f"expected {EXPECTED_DOCUMENT_COUNT} policy files, "
            f"found {len(policy_paths)}"
        )

    if len(documents) != EXPECTED_SECTION_COUNT:
        errors.append(
            f"expected {EXPECTED_SECTION_COUNT} sections, "
            f"found {len(documents)}"
        )

    if not chunks:
        errors.append("no chunks were generated")

    # Generate embeddings.
    try:
        embeddings = embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
    except Exception as exc:
        errors.append(f"embedding generation failed: {exc}")
        embeddings = []

    # Validate embedding count.
    if embeddings and len(embeddings) != len(chunks):
        errors.append(
            f"expected {len(chunks)} embeddings, "
            f"found {len(embeddings)}"
        )

    # Validate each embedding.
    invalid_dimensions = 0
    invalid_values = 0
    empty_embeddings = 0

    for index, embedding in enumerate(embeddings):
        if not embedding:
            empty_embeddings += 1
            errors.append(f"embedding #{index} is empty")
            continue

        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            invalid_dimensions += 1
            errors.append(
                f"embedding #{index} has dimension {len(embedding)}, "
                f"expected {EXPECTED_EMBEDDING_DIMENSION}"
            )

        if not all(isinstance(value, (int, float)) for value in embedding):
            invalid_values += 1
            errors.append(
                f"embedding #{index} contains non-numeric values"
            )

    # Check deterministic behavior on one sample.
    deterministic_pass = False

    if chunks:
        try:
            first_embedding = embedder.embed_query(chunks[0].text)
            second_embedding = embedder.embed_query(chunks[0].text)

            deterministic_pass = first_embedding == second_embedding

            if not deterministic_pass:
                errors.append(
                    "same input produced different embeddings"
                )
        except Exception as exc:
            errors.append(
                f"single-text embedding test failed: {exc}"
            )

    print("Embedding Validation")
    print("--------------------")
    print(
        f"Policy files:       {len(policy_paths)}/{EXPECTED_DOCUMENT_COUNT} "
        f"{'PASS' if len(policy_paths) == EXPECTED_DOCUMENT_COUNT else 'FAIL'}"
    )
    print(
        f"Policy sections:    {len(documents)}/{EXPECTED_SECTION_COUNT} "
        f"{'PASS' if len(documents) == EXPECTED_SECTION_COUNT else 'FAIL'}"
    )
    print(
        f"Chunks:             {len(chunks)} "
        f"{'PASS' if chunks else 'FAIL'}"
    )
    print(
        f"Embeddings:         {len(embeddings)}/{len(chunks)} "
        f"{'PASS' if embeddings and len(embeddings) == len(chunks) else 'FAIL'}"
    )
    print(
        f"Embedding dimension:{' ' if EXPECTED_EMBEDDING_DIMENSION else ''}"
        f"{EXPECTED_EMBEDDING_DIMENSION} "
        f"{'PASS' if embeddings and all(len(e) == EXPECTED_EMBEDDING_DIMENSION for e in embeddings) else 'FAIL'}"
    )
    print(
        f"Numeric values:     "
        f"{'PASS' if embeddings and invalid_values == 0 else 'FAIL'}"
    )
    print(
        f"Deterministic:      "
        f"{'PASS' if deterministic_pass else 'FAIL'}"
    )
    print(
        f"Overall:            "
        f"{'PASS' if not errors else 'FAIL'}"
    )

    if errors:
        print("\nIssues:")
        for error in errors:
            print(f"- {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())