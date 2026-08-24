# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Shared data models for documents, chunks, and RAG responses."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    title: str
    policy_id: str
    version: str
    effective_date: str
    owner: str
    content: str


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    policy_id: str
    section_id: str
    section_title: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    model: str
    retrieval_parameters: dict[str, int | str] = Field(default_factory=dict)
