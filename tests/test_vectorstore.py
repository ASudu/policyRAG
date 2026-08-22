from src.rag.vectorstore import ChromaVectorStore
from src.rag.chunker import Chunk


def test_vectorstore_starts_empty(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    assert store.count() == 0



def test_vectorstore_add_chunks(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        Chunk(
            text="Employees must use approved devices.",
            chunk_id=0,
            document_id="information_security_policy",
            policy_id="IS",
            section_id="IS-001",
            source="information_security_policy.md",
        )
    ]

    embeddings = [[0.1, 0.2, 0.3]]

    store.add_chunks(chunks, embeddings)

    assert store.count() == 1