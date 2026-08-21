"""
This module contains the implementation of the Chunker class, which is responsible for splitting text into smaller chunks for processing. The Chunker class provides methods to define chunk size, overlap, and other parameters to customize the chunking process. It is designed to handle various text formats and can be integrated into larger text processing pipelines.
The Chunker class is particularly useful for preparing text data for machine learning models, natural language processing

Goal: Split text into chunks of a specified size with optional overlap, while preserving the integrity of sentences and paragraphs.

Conceptually, we want to represent each chunk as a Python object with the following structure:
```
Chunk(
    text="...",
    document_id="remote_work_policy",
    policy_id="RW",
    section_id="RW-003",
    source="remote_work_policy.md"
)

```

For our MVP, we do section-based chunking and recursive splitting only if it exceeds the max chunk size. We will not split across sections or paragraphs, and we will not split sentences in half. We will also not split across documents.
"""

from src.rag.loader import Document


class Chunk:
    def __init__(self, text: str, document_id: str, policy_id: str, section_id: str, source: str):
        self.text = text
        self.chunk_id = 0 # if multiple chunks are generated from the same section, this will be incremented (0-indexed)
        self.document_id = document_id
        self.policy_id = policy_id
        self.section_id = section_id
        self.source = source

    def __repr__(self) -> str:
        return f"Chunk(document_id={self.document_id}, policy_id={self.policy_id}, section_id={self.section_id}, source={self.source})"

class Chunker:
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        # Validate parameters
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_chunk_size:
            raise ValueError("overlap must be smaller than max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_text(self, doc:Document) -> list[Chunk]:
        """Split the text into chunks of specified size with optional overlap."""
        text = doc.text
        metadata = doc.metadata
        chunks = []

        if not text.strip():
            return chunks

        start = 0
        previous_chunk_text = None
        while start < len(text):
            # Determine the end of the chunk, ensuring we don't exceed max_chunk_size
            window_end = min(start + self.max_chunk_size, len(text))
            end = window_end
            chunk_text = text[start:end]

            # Ensure we don't split in the middle of a sentence or paragraph
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                split_point = max(last_period, last_newline)
                candidate_end = start + split_point + 1 if split_point != -1 else window_end
                if candidate_end <= start:
                    candidate_end = window_end
                end = candidate_end
                chunk_text = text[start:end]

            # Trim whitespace and check for empty chunks or duplicates
            chunk_text = chunk_text.strip()
            if not chunk_text:
                if end >= len(text):
                    break
                start = max(end, start + 1) # Ensure we make progress even if the chunk is empty
                continue

            # Avoid emitting duplicate chunks in a row, which can happen with overlapping text
            if chunk_text == previous_chunk_text:
                if end >= len(text):
                    break
                start = max(end, start + 1)
                continue

            # If we've made it this far, we can emit the chunk
            emitted_length = len(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=len(chunks), # 0-indexed chunk ID for this section
                document_id=metadata.get("document_id", ""),
                policy_id=metadata.get("policy_id", ""),
                section_id=metadata.get("section_id", ""),
                source=metadata.get("source", "")
            ))
            previous_chunk_text = chunk_text

            if end >= len(text):
                break

            # Move back by overlap for the next chunk while always making progress.
            next_start = max(start + emitted_length - self.overlap, start + 1)
            start = next_start

        return chunks

# ------------------------- TESTING -------------------------
if __name__ == "__main__":
    # Example usage
    doc = Document(
        text="This is a sample policy document. It contains several sentences. "
             "The purpose of this document is to demonstrate chunking. "
             "We will split this text into smaller chunks for processing.",
        metadata={
            "document_id": "sample_policy",
            "policy_id": "SP",
            "section_id": "SP-001",
            "source": "sample_policy.md"
        }
    )

    chunker = Chunker(max_chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(doc)

    for chunk in chunks:
        print(chunk)
        print(f"Chunk text: {chunk.text}\n")