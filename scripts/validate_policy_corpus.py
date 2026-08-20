"""Validate the synthetic Northstar Analytics policy corpus."""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "data" / "documents"
EXPECTED_DOCUMENTS = {
    "leave_policy.md": "LV",
    "remote_work_policy.md": "RW",
    "expense_reimbursement_policy.md": "EX",
    "business_travel_policy.md": "TR",
    "code_of_conduct.md": "COC",
    "information_security_policy.md": "IS",
    "password_and_authentication_policy.md": "PA",
    "data_privacy_policy.md": "DP",
    "employee_onboarding_policy.md": "ON",
    "performance_review_policy.md": "PR",
    "workplace_safety_policy.md": "WS",
    "incident_reporting_policy.md": "IR",
}
REQUIRED_METADATA = {
    "document_id",
    "policy_id",
    "title",
    "version",
    "effective_date",
    "owner",
    "applies_to",
    "status",
}
REQUIRED_SECTIONS = [
    "1. Purpose",
    "2. Scope",
    "3. Definitions",
    "4. Policy Rules",
    "5. Exceptions",
    "6. Procedures",
    "7. Responsibilities",
    "8. References",
]
RULE_PATTERN = re.compile(r"^###\s+([A-Z]+-\d{3})\s+—\s+(.+?)\s*$", re.MULTILINE)
REFERENCE_PATTERN = re.compile(r"\b(?:LV|RW|EX|TR|COC|IS|PA|DP|ON|PR|WS|IR)-\d{3}\b")
NUMBER_PATTERN = re.compile(
    r"(?:\$\s?[0-9][0-9,]*(?:\.[0-9]{1,2})?|[0-9][0-9,]*(?:\.[0-9]+)?%?\s+"
    r"(?:business\s+days?|calendar\s+days?|consecutive\s+business\s+days?|days?|hours?|minutes?|months?|years?|characters?|attempts?|night))"
)


def metadata(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_metadata = False
    for line in text.splitlines():
        if line == "## Document Metadata":
            in_metadata = True
            continue
        if in_metadata and line.startswith("## "):
            break
        if in_metadata and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    failures: list[str] = []
    files = sorted(DOCUMENTS_DIR.glob("*.md"))
    document_results = []
    all_rules: dict[str, str] = {}
    hashes: dict[str, str] = {}

    for filename, expected_prefix in EXPECTED_DOCUMENTS.items():
        path = DOCUMENTS_DIR / filename
        if not path.exists():
            failures.append(f"missing document: {filename}")
            continue
        text = path.read_text(encoding="utf-8")
        values = metadata(text)
        missing_metadata = REQUIRED_METADATA - values.keys()
        missing_sections = [section for section in REQUIRED_SECTIONS if f"## {section}" not in text]
        if missing_metadata:
            failures.append(f"{filename}: missing metadata {sorted(missing_metadata)}")
        if missing_sections:
            failures.append(f"{filename}: missing sections {missing_sections}")
        if values.get("policy_id") != expected_prefix:
            failures.append(f"{filename}: policy_id must be {expected_prefix}")
        if values.get("document_id") != path.stem:
            failures.append(f"{filename}: document_id must be {path.stem}")
        try:
            date.fromisoformat(values["effective_date"])
        except (KeyError, ValueError):
            failures.append(f"{filename}: invalid effective_date")

        rules = RULE_PATTERN.findall(text)
        for rule_id, _title in rules:
            if not rule_id.startswith(f"{expected_prefix}-"):
                failures.append(f"{filename}: rule {rule_id} has the wrong policy prefix")
            if rule_id in all_rules:
                failures.append(f"duplicate rule ID {rule_id} in {filename} and {all_rules[rule_id]}")
            all_rules[rule_id] = filename
        hashes[filename] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        body = text.split("## 4. Policy Rules", 1)[-1].split("## 5. Exceptions", 1)[0]
        numeric_body = re.sub(r"\b[A-Z]+-\d{3}\b", "", body)
        numeric_tokens = re.findall(r"(?<![A-Za-z-])\$?[0-9][0-9,]*(?:\.[0-9]+)?%?", numeric_body)
        malformed_numbers = [token for token in numeric_tokens if not NUMBER_PATTERN.search(body[body.find(token):])]
        if malformed_numbers:
            failures.append(f"{filename}: numeric constraint lacks a recognized unit: {malformed_numbers[0]}")
        document_results.append((filename, values, rules, len(text.split())))

    unexpected = sorted(set(path.name for path in files) - EXPECTED_DOCUMENTS.keys())
    if unexpected:
        failures.append(f"unexpected markdown documents: {unexpected}")
    duplicate_hashes = {digest for digest in hashes.values() if list(hashes.values()).count(digest) > 1}
    if duplicate_hashes:
        failures.append("duplicate document content detected")

    referenced_ids = set()
    for filename, _values, _rules, _words in document_results:
        text = (DOCUMENTS_DIR / filename).read_text(encoding="utf-8")
        referenced_ids.update(REFERENCE_PATTERN.findall(text))
    missing_references = sorted(referenced_ids - all_rules.keys())
    if missing_references:
        failures.append(f"missing referenced rule IDs: {missing_references}")

    metadata_pass = sum(not (REQUIRED_METADATA - values.keys()) for _, values, _, _ in document_results)
    sections_pass = sum(all(f"## {section}" in (DOCUMENTS_DIR / filename).read_text(encoding="utf-8") for section in REQUIRED_SECTIONS) for filename, _, _, _ in document_results)
    rule_count = len(all_rules)
    word_counts = [words for _, _, _, words in document_results]
    word_warning = any(words < 600 or words > 1200 for words in word_counts)

    print("Policy Corpus Validation")
    print("------------------------")
    print(f"Documents:       {len(document_results)}/{len(EXPECTED_DOCUMENTS)} {'PASS' if len(document_results) == len(EXPECTED_DOCUMENTS) else 'FAIL'}")
    print(f"Metadata:        {metadata_pass}/{len(EXPECTED_DOCUMENTS)} {'PASS' if metadata_pass == len(EXPECTED_DOCUMENTS) else 'FAIL'}")
    print(f"Sections:        {sections_pass}/{len(EXPECTED_DOCUMENTS)} {'PASS' if sections_pass == len(EXPECTED_DOCUMENTS) else 'FAIL'}")
    print(f"Rule IDs:        {rule_count} unique {'PASS' if rule_count else 'FAIL'}")
    print(f"References:      {'PASS' if not missing_references else 'FAIL'}")
    print(f"Numeric syntax:  {'PASS' if not failures or not any('numeric constraint' in item for item in failures) else 'FAIL'}")
    print(f"Word counts:     {min(word_counts, default=0)}-{max(word_counts, default=0)} words {'WARN' if word_warning else 'PASS'}")
    print(f"Duplicates:      {'NONE' if not duplicate_hashes else 'FOUND'}")
    print(f"Overall:         {'PASS' if not failures else 'FAIL'}")
    if failures:
        print("\nIssues:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
