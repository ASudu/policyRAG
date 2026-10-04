# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
from src.rag.embeddings import EmbeddingModel


def _fake_embed(*, model, input):
    texts = [input] if isinstance(input, str) else input
    embeddings = []
    for text in texts:
        value = float(len(text))
        embeddings.append([value] * 768)
    return {"embeddings": embeddings}


def test_embed_returns_vector(monkeypatch) -> None:
    monkeypatch.setattr("src.rag.embeddings.ollama.embed", _fake_embed)
    embedder = EmbeddingModel()

    embedding = embedder.embed_query(
        "Employees must use approved devices when working remotely."
    )

    assert isinstance(embedding, list)
    assert len(embedding) == embedder.dimension
    assert all(isinstance(value, float) for value in embedding)


def test_embed_many_returns_vectors(monkeypatch) -> None:
    monkeypatch.setattr("src.rag.embeddings.ollama.embed", _fake_embed)
    embedder = EmbeddingModel()

    texts = [
        "Employees must use approved devices.",
        "Remote work requires manager approval.",
        "Restricted data must remain in approved storage.",
    ]

    embeddings = embedder.embed_documents(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert all(len(embedding) == 768 for embedding in embeddings)
    assert all(
        isinstance(value, float)
        for embedding in embeddings
        for value in embedding
    )


def test_embed_same_text_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr("src.rag.embeddings.ollama.embed", _fake_embed)
    embedder = EmbeddingModel()

    text = "Employees must use approved devices."

    embedding_1 = embedder.embed_query(text)
    embedding_2 = embedder.embed_query(text)

    assert embedding_1 == embedding_2


def test_embed_different_texts_produce_different_vectors(monkeypatch) -> None:
    monkeypatch.setattr("src.rag.embeddings.ollama.embed", _fake_embed)
    embedder = EmbeddingModel()

    embedding_1 = embedder.embed_query(
        "Employees must use approved devices."
    )
    embedding_2 = embedder.embed_query(
        "Employees must submit expenses within ten business days."
    )

    assert embedding_1 != embedding_2