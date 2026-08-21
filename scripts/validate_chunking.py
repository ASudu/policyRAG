"""Validate chunk generation over the policy corpus."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunker import Chunker
from src.rag.loader import Loader


DOCUMENTS_DIR = ROOT / "data" / "documents"
EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_SECTION_COUNT = 92
CHUNK_SIZE = 180
OVERLAP = 20


def main() -> int:
    loader = Loader()
    chunker = Chunker(max_chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    errors = []
    policy_paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    documents = loader.load_corpus(DOCUMENTS_DIR)
    chunks = [chunk for document in documents for chunk in chunker.chunk_text(document)]

    if len(policy_paths) != EXPECTED_DOCUMENT_COUNT:
        errors.append(f"expected {EXPECTED_DOCUMENT_COUNT} policy files, found {len(policy_paths)}")
    if len(documents) != EXPECTED_SECTION_COUNT:
        errors.append(f"expected {EXPECTED_SECTION_COUNT} sections, found {len(documents)}")
    if len(chunks) < len(documents):
        errors.append("chunk count should be at least section count")

    for index, chunk in enumerate(chunks):
        if not chunk.text.strip():
            errors.append(f"chunk #{index} has empty text")
        if len(chunk.text) > CHUNK_SIZE:
            errors.append(f"chunk #{index} exceeds max size ({len(chunk.text)} > {CHUNK_SIZE})")
        if not chunk.document_id or not chunk.policy_id or not chunk.section_id or not chunk.source:
            errors.append(f"chunk #{index} missing metadata")

    # Overlap can legitimately repeat short spans. Only fail on immediate repeats
    # within the same section, which usually indicate non-progressing chunking.
    previous_key = None
    for index, chunk in enumerate(chunks):
        current_key = (chunk.document_id, chunk.policy_id, chunk.section_id, chunk.text)
        if current_key == previous_key:
            errors.append(f"chunk #{index} repeats the previous chunk in the same section")
        previous_key = current_key

    print("Chunking Validation")
    print("-------------------")
    print(f"Policy files:      {len(policy_paths)}/{EXPECTED_DOCUMENT_COUNT} "
          f"{'PASS' if len(policy_paths) == EXPECTED_DOCUMENT_COUNT else 'FAIL'}")
    print(f"Policy sections:   {len(documents)}/{EXPECTED_SECTION_COUNT} "
          f"{'PASS' if len(documents) == EXPECTED_SECTION_COUNT else 'FAIL'}")
    print(f"Generated chunks:  {len(chunks)} PASS")
    print(f"Chunk max size:    {CHUNK_SIZE} "
          f"{'PASS' if all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks) else 'FAIL'}")
    print(f"Metadata present:  {'PASS' if all(chunk.document_id and chunk.policy_id and chunk.section_id and chunk.source for chunk in chunks) else 'FAIL'}")
    print(f"Overall:           {'PASS' if not errors else 'FAIL'}")

    if errors:
        print("\nIssues:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
