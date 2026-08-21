from pathlib import Path

from src.rag.loader import Document, extract_metadata, load_policy_sections


SAMPLE_MARKDOWN = """# Sample Policy

## Document Metadata

- document_id: sample_policy
- policy_id: SP
- title: Sample Policy
- version: 1.0
- effective_date: 2026-01-01
- owner: Test Owner
- applies_to: Test employees
- status: Active

## 1. Purpose

Test purpose.

## 4. Policy Rules

### SP-001 — Sample Rule

Employees must follow the sample rule.

### SP-002 — Another Rule

Managers may approve the sample exception.

## 5. Exceptions

Documented exceptions are allowed.
"""


def test_load_policy_sections_from_sample_file(tmp_path: Path) -> None:
    policy_path = tmp_path / "sample_policy.md"
    policy_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

    documents = load_policy_sections(policy_path)

    assert len(documents) == 2
    assert all(isinstance(document, Document) for document in documents)
    assert documents[0].metadata == {
        "document_id": "sample_policy",
        "policy_id": "SP",
        "section_id": "SP-001",
        "source": "sample_policy.md",
    }
    assert documents[0].text == "Sample Rule\n\nEmployees must follow the sample rule."
    assert documents[1].metadata["section_id"] == "SP-002"
    assert "Managers may approve" in documents[1].text


def test_extract_metadata_normalizes_keys() -> None:
    metadata = extract_metadata(SAMPLE_MARKDOWN)

    assert metadata["document_id"] == "sample_policy"
    assert metadata["policy_id"] == "SP"
    assert metadata["effective_date"] == "2026-01-01"
    assert metadata["applies_to"] == "Test employees"


def test_load_all_policy_documents() -> None:
    documents_dir = Path(__file__).parents[1] / "data" / "documents"
    documents = [
        document
        for policy_path in sorted(documents_dir.glob("*.md"))
        for document in load_policy_sections(policy_path)
    ]

    assert len(list(documents_dir.glob("*.md"))) == 12
    assert len(documents) == 92
    assert all(document.text for document in documents)
    assert all(document.metadata["section_id"].count("-") == 1 for document in documents)
