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

import os
from pathlib import Path
import re

# Define the Document class to represent each document with text and metadata
class Document:
    """A simple representation of a document with text and metadata."""
    def __init__(self, text: str, metadata: dict[str, str]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Document(metadata={self.metadata})"

# Pattern to match policy rule sections
RULE_PATTERN = re.compile(
    r"^###\s+([A-Z]+-\d{3})\s+—\s+(.+?)\s*$",
    re.MULTILINE,
)

# ------------------------- HELPER FUNCTIONS -------------------------
def extract_metadata(markdown: str) -> dict[str, str]:
    """Extracts metadata from the markdown content."""
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

def load_policy_sections(path: Path) -> list[Document]:
    """Loads policy sections from a markdown file and returns a list of Document objects."""
    markdown = path.read_text(encoding="utf-8")
    metadata = extract_metadata(markdown)

    matches = list(RULE_PATTERN.finditer(markdown))
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

# ------------------------- TESTING -------------------------
if __name__ == "__main__":
    # Example usage: Load all markdown files in the "data/documents" directory
    documents = []
    for path in Path("data/documents").glob("*.md"):
        documents.extend(load_policy_sections(path))

    # Print the first two loaded documents for verification
    for document in documents[:2]:
        print(document)
        print(document.text[:100])  # Print the first 100 characters of the text
        print(document.metadata)
        print("-" * 40)