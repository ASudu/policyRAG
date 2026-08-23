import json

from src.evaluation.claims import extract_claims


def test_extract_claims_returns_empty_for_blank_answer() -> None:
    assert extract_claims("   ") == []


def test_extract_claims_parses_structured_json(monkeypatch) -> None:
    def fake_chat(**kwargs):
        assert "format" in kwargs
        assert kwargs["options"]["temperature"] == 0
        return {
            "message": {
                "content": json.dumps(
                    {
                        "claims": [
                            "MFA is required for remote access.",
                            "MFA is required for remote access.",
                            "",
                            7,
                            "Privileged accounts require MFA.",
                        ]
                    }
                )
            }
        }

    monkeypatch.setattr("src.evaluation.claims.chat", fake_chat)

    claims = extract_claims("Yes. MFA is required for remote access and privileged accounts.")

    assert claims == [
        "MFA is required for remote access.",
        "Privileged accounts require MFA.",
    ]


def test_extract_claims_returns_empty_on_invalid_json(monkeypatch) -> None:
    def fake_chat(**kwargs):
        return {"message": {"content": "not json"}}

    monkeypatch.setattr("src.evaluation.claims.chat", fake_chat)

    assert extract_claims("Employees must submit expenses within 10 business days.") == []
