"""Validate loading of the Markdown policy corpus."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.loader import load_policy_sections


DOCUMENTS_DIR = ROOT / "data" / "documents"
EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_SECTION_COUNT = 92
REQUIRED_METADATA = {"document_id", "policy_id", "section_id", "source"}


def main() -> int:
    policy_paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    loaded_documents = []
    errors = []

    for policy_path in policy_paths:
        try:
            loaded_documents.extend(load_policy_sections(policy_path))
        except Exception as error:  # Report the file that failed to load.
            errors.append(f"{policy_path.name}: {error}")

    section_ids = [document.metadata.get("section_id") for document in loaded_documents]
    if len(policy_paths) != EXPECTED_DOCUMENT_COUNT:
        errors.append(f"expected {EXPECTED_DOCUMENT_COUNT} documents, found {len(policy_paths)}")
    if len(loaded_documents) != EXPECTED_SECTION_COUNT:
        errors.append(f"expected {EXPECTED_SECTION_COUNT} sections, found {len(loaded_documents)}")
    if len(section_ids) != len(set(section_ids)):
        errors.append("duplicate section IDs found")

    for document in loaded_documents:
        missing_metadata = REQUIRED_METADATA - document.metadata.keys()
        if missing_metadata:
            errors.append(
                f"{document.metadata.get('source', '<unknown>')}: "
                f"missing metadata {sorted(missing_metadata)}"
            )
        if not document.text.strip():
            errors.append(f"{document.metadata.get('section_id', '<unknown>')}: empty text")

    print("Loader Validation")
    print("-----------------")
    print(f"Policy files:     {len(policy_paths)}/{EXPECTED_DOCUMENT_COUNT} "
          f"{'PASS' if len(policy_paths) == EXPECTED_DOCUMENT_COUNT else 'FAIL'}")
    print(f"Loaded sections:  {len(loaded_documents)}/{EXPECTED_SECTION_COUNT} "
          f"{'PASS' if len(loaded_documents) == EXPECTED_SECTION_COUNT else 'FAIL'}")
    print(f"Metadata/text:    {'PASS' if not errors else 'FAIL'}")
    print(f"Unique section IDs: {'PASS' if len(section_ids) == len(set(section_ids)) else 'FAIL'}")
    print(f"Overall:          {'PASS' if not errors else 'FAIL'}")

    if errors:
        print("\nIssues:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
