# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""Tests for deterministic policy checks."""

from src.evaluation.hard_checks import check_polarity


def test_same_polarity_passes():
	result = check_polarity(
		"Employees may work remotely from their approved home state.",
		"Employees are allowed to work remotely from their approved home state.",
	)

	assert result.passed


def test_permission_reversal_fails():
	result = check_polarity(
		"Employees may work remotely from another state.",
		"Employees may not work remotely from another state.",
	)

	assert not result.passed


def test_requirement_reversal_fails():
	result = check_polarity(
		"Employees must submit the report within 10 business days.",
		"Employees are not required to submit the report within 10 business days.",
	)

	assert not result.passed
