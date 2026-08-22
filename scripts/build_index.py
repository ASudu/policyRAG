"""Build the local policy vector index."""

"""Index policy chunks into ChromaDB."""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunker import Chunker
from src.rag.embeddings import EmbeddingModel
from src.rag.loader import Loader
from src.rag.vectorstore import ChromaVectorStore
load_dotenv()

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", ROOT / "data" / "documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "180"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def main() -> None:
    # Instantiate loader, chunker and embedder
    print(f"Loading documents from: {DOCUMENTS_DIR}")
    loader = Loader()
    chunker = Chunker(
        max_chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    embedder = EmbeddingModel(
        model_name=EMBEDDING_MODEL
    )

    # Instantiate vector store
    print(f"Persisting ChromaDB to: {CHROMA_DIR}")
    vectorstore = ChromaVectorStore(
        persist_directory=CHROMA_DIR
    )

    # Load documents
    documents = loader.load_corpus(DOCUMENTS_DIR)

    # Chunk the documents
    chunks = [
        chunk
        for document in documents
        for chunk in chunker.chunk_text(document)
    ]

    print(f"Loaded sections: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    embeddings = embedder.embed_documents(
        [chunk.text for chunk in chunks]
    )

    print(f"Generated embeddings: {len(embeddings)}")

    # Add chunks and embeddings to the vector store
    vectorstore.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"Chroma records: {vectorstore.count()}")


if __name__ == "__main__":
    main()
