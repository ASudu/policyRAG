"""Validate the synthetic SME-review dataset against the policy corpus."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "golden_qa.jsonl"
DOCS = ROOT / "data" / "documents"
RULE_RE = re.compile(r"^###\s+([A-Z]+-\d{3})\s+—\s+(.+?)\s*$", re.M)
CATEGORIES = {
    "simple_factual": 8,
    "multi_part": 8,
    "exception_based": 8,
    "numerical": 7,
    "negative_prohibition": 4,
    "multi_hop": 4,
    "ambiguous_adversarial": 4,
    "out_of_scope": 3,
}
FAILURES = {
    "INCOMPLETE_ANSWER", "INCORRECT_INFORMATION", "WRONG_NUMERICAL_VALUE",
    "MISSING_EXCEPTION", "UNSUPPORTED_CLAIM", "WRONG_POLICY_REFERENCE",
    "CONTRADICTORY_INFORMATION", "WRONG_POLICY_INTERPRETATION",
    "IRRELEVANT_RETRIEVAL", "MISSING_REQUIRED_APPROVAL",
}


def load_rules() -> dict[str, dict[str, str]]:
    rules = {}
    for path in DOCS.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        matches = list(RULE_RE.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else text.find("\n## 5. Exceptions", match.end())
            body = text[match.end():end].strip()
            rules[match.group(1)] = {
                "document_id": path.stem,
                "text": f"### {match.group(1)} — {match.group(2)}\n\n{body}",
            }
    return rules


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    rules = load_rules()
    if not DATASET.exists():
        print(f"Missing dataset: {DATASET}")
        return 1
    records = []
    with DATASET.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                fail(errors, f"line {line_number}: invalid JSON ({exc.msg})")

    ids = [record.get("id") for record in records]
    questions = [record.get("question") for record in records]
    if len(ids) != len(set(ids)):
        fail(errors, "duplicate record IDs")
    if len(questions) != len(set(questions)):
        fail(errors, "duplicate questions")

    decisions = Counter()
    categories = Counter()
    failures = Counter()
    obligations_answerable = 0
    obligations_not_answerable = 0
    retrieval_failures = 0
    generation_failures = 0

    for record in records:
        record_id = record.get("id", "<missing-id>")
        required = {"id", "question", "ai_response", "retrieved_chunks", "sme_review", "certified_answer", "certified_claims", "answer_obligations", "authoritative_evidence", "category", "difficulty"}
        missing = required - record.keys()
        if missing:
            fail(errors, f"{record_id}: missing fields {sorted(missing)}")
            continue
        if not isinstance(record["question"], str) or not record["question"].strip():
            fail(errors, f"{record_id}: empty question")
        if not isinstance(record["ai_response"], str) or not record["ai_response"].strip():
            fail(errors, f"{record_id}: empty AI response")
        if not isinstance(record["retrieved_chunks"], list) or not record["retrieved_chunks"]:
            fail(errors, f"{record_id}: retrieved_chunks must be non-empty")
        if not isinstance(record["authoritative_evidence"], list):
            fail(errors, f"{record_id}: authoritative_evidence must be a list")
        for rule_id in record["authoritative_evidence"]:
            if rule_id not in rules:
                fail(errors, f"{record_id}: unknown authoritative rule {rule_id}")
        retrieved_ids = []
        for retrieved in record["retrieved_chunks"]:
            for field in ("chunk_id", "document_id", "section_id", "text", "relevance"):
                if field not in retrieved:
                    fail(errors, f"{record_id}: retrieved chunk missing {field}")
            rule_id = retrieved.get("section_id")
            retrieved_ids.append(rule_id)
            if rule_id not in rules:
                fail(errors, f"{record_id}: unknown retrieved rule {rule_id}")
            elif retrieved.get("chunk_id") != rule_id or retrieved.get("document_id") != rules[rule_id]["document_id"] or retrieved.get("text") != rules[rule_id]["text"]:
                fail(errors, f"{record_id}: retrieved chunk {rule_id} is not traceable to corpus text")
        review = record["sme_review"]
        decision = review.get("decision") if isinstance(review, dict) else None
        decisions[decision] += 1
        if decision not in {"THUMBS_UP", "THUMBS_DOWN"}:
            fail(errors, f"{record_id}: invalid SME decision {decision}")
        if decision == "THUMBS_DOWN":
            if not review.get("feedback") or not review.get("primary_failure"):
                fail(errors, f"{record_id}: thumbs-down requires feedback and primary_failure")
            if review.get("primary_failure") not in FAILURES:
                fail(errors, f"{record_id}: invalid primary_failure")
            failures[review.get("primary_failure")] += 1
        elif review.get("feedback") is not None:
            fail(errors, f"{record_id}: thumbs-up feedback should be null")
        categories[record["category"]] += 1

        if not isinstance(record["certified_answer"], str) or not record["certified_answer"].strip():
            fail(errors, f"{record_id}: missing certified answer")
        for claim in record["certified_claims"]:
            supporting = claim.get("supporting_evidence")
            if not claim.get("claim") or not isinstance(supporting, list) or not supporting:
                fail(errors, f"{record_id}: certified claim lacks supporting evidence")
            elif not set(supporting).issubset(set(record["authoritative_evidence"])):
                fail(errors, f"{record_id}: claim evidence is outside authoritative_evidence")
            for rule_id in supporting or []:
                if rule_id not in rules:
                    fail(errors, f"{record_id}: claim cites unknown rule {rule_id}")
        for obligation in record["answer_obligations"]:
            status = obligation.get("evidence_status")
            if status not in {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "NOT_ANSWERABLE"}:
                fail(errors, f"{record_id}: invalid obligation status {status}")
            if status == "NOT_ANSWERABLE":
                obligations_not_answerable += 1
                if obligation.get("supporting_claims") or obligation.get("supporting_evidence"):
                    fail(errors, f"{record_id}: NOT_ANSWERABLE obligation has fabricated support")
            else:
                obligations_answerable += 1
                if not obligation.get("supporting_claims") or not obligation.get("supporting_evidence"):
                    fail(errors, f"{record_id}: answerable obligation lacks support")
                if not set(obligation.get("supporting_evidence", [])).issubset(set(record["authoritative_evidence"])):
                    fail(errors, f"{record_id}: obligation evidence is outside authoritative_evidence")

        if decision == "THUMBS_DOWN":
            authoritative = set(record["authoritative_evidence"])
            retrieved = set(retrieved_ids)
            if authoritative != retrieved:
                retrieval_failures += 1
            if authoritative and authoritative.issubset(retrieved):
                generation_failures += 1

    for category, minimum in CATEGORIES.items():
        if categories[category] < minimum:
            fail(errors, f"category {category}: {categories[category]} records, expected at least {minimum}")
    if not 45 <= decisions["THUMBS_UP"] <= 50:
        fail(errors, f"THUMBS_UP count {decisions['THUMBS_UP']} outside 45-50")
    if not 10 <= decisions["THUMBS_DOWN"] <= 15:
        fail(errors, f"THUMBS_DOWN count {decisions['THUMBS_DOWN']} outside 10-15")
    if retrieval_failures < 3:
        fail(errors, f"only {retrieval_failures} thumbs-down retrieval-failure examples; need at least 3")
    if generation_failures < 3:
        fail(errors, f"only {generation_failures} thumbs-down generation-failure examples; need at least 3")

    print("Golden QA Validation")
    print("--------------------")
    print(f"Records:           {len(records)} {'PASS' if len(records) == 60 else 'FAIL'}")
    print(f"THUMBS_UP:         {decisions['THUMBS_UP']} {'PASS' if 45 <= decisions['THUMBS_UP'] <= 50 else 'FAIL'}")
    print(f"THUMBS_DOWN:       {decisions['THUMBS_DOWN']} {'PASS' if 10 <= decisions['THUMBS_DOWN'] <= 15 else 'FAIL'}")
    print(f"Answerable:        {obligations_answerable}")
    print(f"Not answerable:    {obligations_not_answerable}")
    print(f"Categories:        {'PASS' if not any(categories[c] < minimum for c, minimum in CATEGORIES.items()) else 'FAIL'}")
    print(f"Retrieval failures:{retrieval_failures} {'PASS' if retrieval_failures >= 3 else 'FAIL'}")
    print(f"Generation failures:{generation_failures} {'PASS' if generation_failures >= 3 else 'FAIL'}")
    print(f"Failure categories:{dict(failures)}")
    print(f"Overall:           {'PASS' if not errors else 'FAIL'}")
    if errors:
        print("\nIssues:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
