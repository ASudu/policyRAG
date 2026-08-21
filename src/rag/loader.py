"""
This module contains the implementation of the Loader class, which is responsible for loading data from various sources and formats.
The Loader class provides methods to read data from files, databases, and APIs, and convert it into a standardized format for further processing.
It also includes error handling and logging capabilities to ensure smooth data loading operations.

Goal: Turn each .md file into Python objects containing text + metadata.

Conceptually, we want to represent each document as a Python object with the following structure:
```
Document(
    text="Employees may work remotely...",
    metadata={
        "document_id": "remote_work_policy",
        "policy_id": "RW",
        "section_id": "RW-003",
        "source": "remote_work_policy.md"
    }
)
```
"""
from dotenv import load_dotenv
from pathlib import Path
import re
import os

load_dotenv()  # Load environment variables from .env file

# Define the Document class to represent each document with text and metadata
class Document:
    """A simple representation of a document with text and metadata."""
    def __init__(self, text: str, metadata: dict[str, str]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Document(metadata={self.metadata})"

class Loader:
    """Loader for policy markdown files."""

    RULE_PATTERN = re.compile(
        r"^###\s+([A-Z]+-\d{3})\s+—\s+(.+?)\s*$",
        re.MULTILINE,
    )

    def extract_metadata(self, markdown: str) -> dict[str, str]:
        """Extract metadata from the `## Document Metadata` section."""
        metadata = {}
        in_metadata = False

        for line in markdown.splitlines():
            if line.strip() == "## Document Metadata":
                in_metadata = True
                continue

            if in_metadata and line.startswith("## "):
                break

            if in_metadata and line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                metadata[key.strip().lower().replace(" ", "_")] = value.strip()

        return metadata

    def load_policy_sections(self, path: Path) -> list[Document]:
        """Load rule sections from a single markdown policy file."""
        markdown = path.read_text(encoding="utf-8")
        metadata = self.extract_metadata(markdown)

        matches = list(self.RULE_PATTERN.finditer(markdown))
        documents = []

        for index, match in enumerate(matches):
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else markdown.find("\n## 5. Exceptions", start)
            )

            rule_text = markdown[start:end].strip()
            section_id = match.group(1)

            section_metadata = {
                "document_id": metadata["document_id"],
                "policy_id": metadata["policy_id"],
                "section_id": section_id,
                "source": path.name,
            }

            text = f"{match.group(2)}\n\n{rule_text}"
            documents.append(Document(text, section_metadata))

        return documents

    def load_corpus(self, documents_dir: Path) -> list[Document]:
        """Load all markdown policy documents from a directory."""
        documents = []
        for path in sorted(documents_dir.glob("*.md")):
            documents.extend(self.load_policy_sections(path))
        return documents

# ------------------------- TESTING -------------------------
if __name__ == "__main__":
    loader = Loader()
    documents = loader.load_corpus(Path(os.getenv("DOCUMENTS_DIR", "data/documents")))

    for document in documents[:2]:
        print(document)
        print(document.text[:100])
        print(document.metadata)
        print("-" * 40)