# tests/eval/test_metrics.py
import json

from hippofloop.eval.evaluator import _SCHEMAS
from hippofloop.eval.metrics import (
    field_accuracy,
    json_validity,
    parse_model_output,
    schema_validity,
)
from hippofloop.protocols import Task

# -- JSON validity --

def test_json_validity_valid():
    assert json_validity('{"summary": "test", "tone": "neutral"}') is True


def test_json_validity_invalid():
    assert json_validity("not json at all") is False


def test_json_validity_empty():
    assert json_validity("") is False


def test_json_validity_with_markdown_fences():
    """Models sometimes wrap JSON in markdown code fences."""
    assert json_validity('```json\n{"summary": "test"}\n```') is True


def test_parse_model_output_strips_fences():
    result = parse_model_output('```json\n{"key": "val"}\n```')
    assert result == {"key": "val"}


def test_parse_model_output_returns_none_for_invalid():
    assert parse_model_output("not json") is None


# -- Schema validity --

SUMMARIZE_SCHEMA = _SCHEMAS[Task.SUMMARIZE]

def test_schema_validity_valid():
    output = json.dumps({
        "summary": "test", "tone": "neutral", "phase": "opening",
        "pattern": "debugging", "key_moments": [], "open_threads": [],
    })
    assert schema_validity(output, SUMMARIZE_SCHEMA) is True


def test_schema_validity_missing_field():
    output = json.dumps({"summary": "test", "tone": "neutral"})
    assert schema_validity(output, SUMMARIZE_SCHEMA) is False


def test_schema_validity_wrong_type():
    output = json.dumps({
        "summary": "test", "tone": "neutral", "phase": "opening",
        "pattern": "debugging", "key_moments": "not a list", "open_threads": [],
    })
    assert schema_validity(output, SUMMARIZE_SCHEMA) is False


def test_schema_validity_not_json():
    assert schema_validity("not json", SUMMARIZE_SCHEMA) is False


# -- Field accuracy --

def test_field_accuracy_exact_match():
    predicted = {"kind": "directive", "scope": "universal"}
    ground_truth = {"kind": "directive", "scope": "universal"}
    result = field_accuracy(predicted, ground_truth, exact_fields=["kind", "scope"])
    assert result == {"kind": 1.0, "scope": 1.0}


def test_field_accuracy_mismatch():
    predicted = {"kind": "preference", "scope": "project"}
    ground_truth = {"kind": "directive", "scope": "universal"}
    result = field_accuracy(predicted, ground_truth, exact_fields=["kind", "scope"])
    assert result == {"kind": 0.0, "scope": 0.0}


def test_field_accuracy_numeric_within_threshold():
    predicted = {"importance": 0.82}
    ground_truth = {"importance": 0.85}
    result = field_accuracy(
        predicted, ground_truth,
        numeric_fields=["importance"], threshold=0.15,
    )
    assert result == {"importance": 1.0}


def test_field_accuracy_numeric_outside_threshold():
    predicted = {"importance": 0.5}
    ground_truth = {"importance": 0.85}
    result = field_accuracy(
        predicted, ground_truth,
        numeric_fields=["importance"], threshold=0.15,
    )
    assert result == {"importance": 0.0}


def test_field_accuracy_missing_field():
    predicted = {"kind": "directive"}
    ground_truth = {"kind": "directive", "scope": "universal"}
    result = field_accuracy(predicted, ground_truth, exact_fields=["kind", "scope"])
    assert result["kind"] == 1.0
    assert result["scope"] == 0.0
