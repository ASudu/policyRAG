from pathlib import Path

import pytest

from src.rag.chunker import Chunk, Chunker
from src.rag.loader import Document, Loader


def test_chunk_text_returns_chunk_objects() -> None:
    doc = Document(
        text=(
            "Remote work requires manager coordination. "
            "Employees must use approved devices. "
            "Restricted data must remain in approved storage."
        ),
        metadata={
            "document_id": "remote_work_policy",
            "policy_id": "RW",
            "section_id": "RW-007",
            "source": "remote_work_policy.md",
        },
    )

    chunks = Chunker(max_chunk_size=70, overlap=10).chunk_text(doc)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.document_id == "remote_work_policy" for chunk in chunks)
    assert all(chunk.policy_id == "RW" for chunk in chunks)
    assert all(chunk.section_id == "RW-007" for chunk in chunks)
    assert all(chunk.source == "remote_work_policy.md" for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunker_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        Chunker(max_chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        Chunker(max_chunk_size=50, overlap=-1)
    with pytest.raises(ValueError):
        Chunker(max_chunk_size=50, overlap=50)


def test_chunk_text_short_input_no_infinite_loop() -> None:
    doc = Document(
        text="Short text.",
        metadata={
            "document_id": "sample_policy",
            "policy_id": "SP",
            "section_id": "SP-001",
            "source": "sample_policy.md",
        },
    )

    chunks = Chunker(max_chunk_size=500, overlap=50).chunk_text(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunk_all_loaded_policy_sections() -> None:
    loader = Loader()
    chunker = Chunker(max_chunk_size=180, overlap=20)
    documents_dir = Path(__file__).parents[1] / "data" / "documents"

    docs = loader.load_corpus(documents_dir)
    chunked = [chunk for doc in docs for chunk in chunker.chunk_text(doc)]

    assert len(docs) == 92
    assert len(chunked) >= len(docs)
    assert all(chunk.text.strip() for chunk in chunked)
    assert all(len(chunk.text) <= 180 for chunk in chunked)
