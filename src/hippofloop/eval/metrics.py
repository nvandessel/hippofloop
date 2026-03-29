"""Evaluation metrics for HippoFloop model output quality."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_model_output(output: str) -> dict | None:
    """Parse model output as JSON, stripping markdown fences if present.

    Returns the parsed dict, or None if the output is not valid JSON.
    This is the single entry point for parsing model responses — use it
    instead of calling json.loads directly.
    """
    text = _strip_markdown_fences(output)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def json_validity(output: str) -> bool:
    """Check if output is a valid JSON object (dict). Handles markdown code fences."""
    return parse_model_output(output) is not None


def schema_validity(output: str, schema: dict[str, Any]) -> bool:
    """Check if output matches expected schema (required fields + types)."""
    parsed = parse_model_output(output)
    if parsed is None:
        return False

    required = schema.get("required_fields", [])
    field_types = schema.get("field_types", {})

    for field_name in required:
        if field_name not in parsed:
            return False

    for field_name, expected_type in field_types.items():
        if field_name in parsed and not isinstance(parsed[field_name], expected_type):
            return False

    return True


def field_accuracy(
    predicted: dict[str, Any],
    ground_truth: dict[str, Any],
    exact_fields: list[str] | None = None,
    numeric_fields: list[str] | None = None,
    threshold: float = 0.15,
) -> dict[str, float]:
    """Compare predicted fields against ground truth.

    exact_fields: Fields compared by equality (1.0 if match, 0.0 if not).
    numeric_fields: Fields compared within threshold (1.0 if within, 0.0 if not).
    Missing fields in predicted score 0.0.
    """
    result: dict[str, float] = {}

    for field_name in exact_fields or []:
        pred_val = predicted.get(field_name)
        true_val = ground_truth.get(field_name)
        result[field_name] = 1.0 if pred_val == true_val else 0.0

    for field_name in numeric_fields or []:
        pred_val = predicted.get(field_name)
        true_val = ground_truth.get(field_name)
        if pred_val is None or true_val is None:
            result[field_name] = 0.0
        else:
            result[field_name] = 1.0 if abs(pred_val - true_val) <= threshold else 0.0

    return result


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
