"""
In this module, we define the embedding classes used for the RAG model. These embeddings are responsible for converting input text into dense vector representations that can be used by the model for various tasks such as retrieval and generation. The embeddings can be based on pre-trained models or custom-trained embeddings, depending on the specific use case.

Why embed_query() and embed_documents() methods are used:
- The embed_query() method is used during retrieval to embed a single chunk of text (query) into a dense vector representation. 
- The embed_documents() method is used during indexing to embed multiple chunks.
"""

import os
from dotenv import load_dotenv
import ollama


class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-embed-text", dimension: int = 768) -> None:
        self.model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        """To embed a chunk for retrieval

        Args:
            text (str): Chunk to embed

        Returns:
            list[float]: The embedding vector for the input text.
        """
        response = ollama.embed(
            model=self.model_name,
            input=text,
        )

        return response["embeddings"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """To embed multiple chunks for indexing

        Args:
            texts (list[str]): List of chunks to embed

        Returns:
            list[list[float]]: List of embedding vectors for the input texts.
        """
        response = ollama.embed(
            model=self.model_name,
            input=texts,
        )

        return response["embeddings"]

# ----------------- TESTING -----------------
if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    # Initialize the embedding model
    embedding_model = EmbeddingModel()

    # Test embedding a single query
    query_text = "What is the policy on data privacy?"
    query_embedding = embedding_model.embed_query(query_text)
    print(f"Query Embedding (length {len(query_embedding)}): {query_embedding}")

    # Test embedding multiple documents
    document_texts = [
        "Data privacy is a critical aspect of our operations.",
        "We ensure that all user data is handled securely.",
        "Our policies comply with international data protection regulations."
    ]
    document_embeddings = embedding_model.embed_documents(document_texts)
    for i, emb in enumerate(document_embeddings):
        print(f"Document {i} Embedding (length {len(emb)}): {emb}")