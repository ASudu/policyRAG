"""
In this module, we define the `Retriever` class, which is responsible for retrieving relevant documents or information based on a given query. The `Retriever` class utilizes various retrieval techniques and algorithms to efficiently search through a large corpus of data and return the most relevant results.
"""

from dataclasses import dataclass

from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import ChromaVectorStore


@dataclass
class RetrievedChunk:
    text: str
    chunk_id: int
    distance: float
    document_id: str
    policy_id: str
    section_id: str
    source: str


class Retriever:
    def __init__(self, embedder: EmbeddingModel, vector_store: ChromaVectorStore,) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5,) -> list[RetrievedChunk]:

        # Parametr checks
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Query the top-k results from the vector store
        results = self.vector_store.query(
            query_embedding,
            top_k=top_k,
        )

        # Process retrieved chunks
        retrieved_chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Iterate through the retrieved results and create RetrievedChunk objects
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                RetrievedChunk(
                    text=document,
                    distance=distance,
                    chunk_id=metadata["chunk_id"],
                    document_id=metadata["document_id"],
                    policy_id=metadata["policy_id"],
                    section_id=metadata["section_id"],
                    source=metadata["source"],
                )
            )

        return retrieved_chunks