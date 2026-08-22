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