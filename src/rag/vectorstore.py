# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
In this module, we define the chroma db structure to store our embeddings and metadata. So, our pipeline gives two things:

Chunk(
    text: str,
    document_id: str,
    policy_id: str,
    section_id: str,
    source: str,
)

and

Embedding(
    vector: List[float],
)

So, we need to combine these two into a chroma record like this:
Chroma record
├── id
├── document       ← chunk.text
├── embedding      ← 768-dimensional vector
└── metadata
      ├── document_id
      ├── policy_id
      ├── section_id
      └── source
"""

import os
from pathlib import Path
import chromadb
from src.rag.chunker import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str | Path = "data/chroma", collection_name: str = "policy_chunks",) -> None:
        """
        Initialize the ChromaVectorStore.

        Args:
            persist_directory (str | Path): The directory to persist the Chroma database.
            collection_name (str): The name of the collection to use.
        """

        self.persist_directory = str(persist_directory) # Convert Path to str if necessary
        # Ensure the persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize the Chroma client and collection
        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Get or create the collection with the specified name and metadata
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]],) -> None:
        """
        Add a list of chunks and their corresponding embeddings to the vector store.
        """

        # Check that the number of chunks matches the number of embeddings
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        # Aggregate the data into the format required by Chroma
        ids = [
            f"{chunk.document_id}:{chunk.section_id}:{index}"
            for index, chunk in enumerate(chunks)
        ]
        documents = [chunk.text for chunk in chunks]
        metadata = [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "policy_id": chunk.policy_id,
                "section_id": chunk.section_id,
                "source": chunk.source,
            }
            for chunk in chunks
        ]

        # Add the record to the collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

    def query(self, embedding: list[float], top_k: int = 5,) -> dict:
        """
        Query the vector store for the top_k most similar chunks to the given embedding.
        """
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """
        Number of records in the collection.
        """
        return self.collection.count()

    def reset(self) -> None:
        """
        Reset the collection by deleting all records.
        """
        self.client.delete_collection(
            name=self.collection.name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

# ------------ TESTING ------------
if __name__ == "__main__":
    # Temp dir
    temp_dir = Path("data/chroma_test")
    os.makedirs(temp_dir, exist_ok=True)
    # Example usage
    vector_store = ChromaVectorStore(persist_directory=temp_dir, collection_name="test_chunks")

    # Create some dummy chunks and embeddings
    chunks = [
        Chunk(text="This is the first chunk.", chunk_id=0, document_id="doc1", policy_id="policy1", section_id="sec1", source="source1"),
        Chunk(text="This is the second chunk.", chunk_id=1, document_id="doc2", policy_id="policy2", section_id="sec2", source="source2"),
    ]
    embeddings = [
        [0.1] * 768,  # Dummy embedding for the first chunk
        [0.3] * 768,  # Dummy embedding for the second chunk
    ]

    # Add chunks to the vector store
    vector_store.add_chunks(chunks, embeddings)

    # Query the vector store with a dummy embedding
    query_embedding = [0.15] * 768
    results = vector_store.query(query_embedding, top_k=2)
    print("Query Results:", results)

    # Count the number of records in the collection
    print("Number of records in the collection:", vector_store.count())

    # Reset the collection
    vector_store.reset()
    print("Collection reset. Number of records now:", vector_store.count())