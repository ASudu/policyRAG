import os
from pathlib import Path
from dotenv import load_dotenv

from src.rag.embeddings import EmbeddingModel
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.vectorstore import ChromaVectorStore

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", ROOT / "data" / "chroma")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2:3b")


def main() -> None:
    # Instantiate the components for the pipeline
    embedder = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIR
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    generator = Generator(
        model_name=GENERATION_MODEL
    )

    print("Retrieval and Generation pipeline initialized successfully.")

    # Sample question to test the retrieval and generation pipeline
    question = "What are the rules for remote work?"

    print("\nQuestion:")
    print(question)

    # Retrieve relevant chunks from vector store
    chunks = retriever.retrieve(
        query=question,
        top_k=5, # top 5 relevant chunks 
    )

    print("\nRetrieved chunks:")
    for chunk in chunks:
        print(
            f"- {chunk.policy_id} / "
            f"{chunk.section_id} "
            f"(distance={chunk.distance:.4f})"
        )

    # Generate an answer based on the retrieved chunks
    answer = generator.generate(
        query=question,
        chunks=chunks,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()