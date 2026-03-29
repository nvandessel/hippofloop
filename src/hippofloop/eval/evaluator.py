"""Model evaluator — run model on test data, compute quality metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from hippofloop.eval.metrics import (
    field_accuracy,
    json_validity,
    parse_model_output,
    schema_validity,
)
from hippofloop.protocols import EvalResult, SFTPair, Task

logger = logging.getLogger(__name__)

# Schema definitions per task
_SCHEMAS: dict[str, dict[str, Any]] = {
    Task.SUMMARIZE: {
        "required_fields": ["summary", "tone", "phase", "pattern", "key_moments", "open_threads"],
        "field_types": {
            "summary": str, "tone": str, "phase": str, "pattern": str,
            "key_moments": list, "open_threads": list,
        },
    },
    Task.ARC: {
        "required_fields": ["arc", "dominant_tone", "session_outcome", "themes"],
        "field_types": {
            "arc": str, "dominant_tone": str, "session_outcome": str, "themes": list,
        },
    },
    Task.EXTRACT: {
        "required_fields": ["candidates"],
        "field_types": {"candidates": list},
    },
    Task.CLASSIFY: {
        "required_fields": ["classified"],
        "field_types": {"classified": list},
    },
    Task.RELATE: {
        "required_fields": ["relationships"],
        "field_types": {"relationships": list},
    },
}


class ModelEvaluator:
    """Evaluates a model's output quality against ground-truth SFT pairs.

    Takes a model_fn callable (system, user) -> response string.
    This allows testing with mocks, local GGUF models, or API models.
    """

    def __init__(self, model_fn: Callable[[str, str], str]) -> None:
        self._model_fn = model_fn

    def evaluate(self, test_data: list[SFTPair]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for pair in test_data:
            result = self._evaluate_one(pair)
            results.append(result)
        return results

    def summary_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Compute aggregate metrics from evaluation results."""
        total = len(results)
        if total == 0:
            return {"total": 0}

        json_valid_count = sum(1 for r in results if r.json_valid)
        schema_valid_count = sum(1 for r in results if r.schema_valid)

        report: dict[str, Any] = {
            "total": total,
            "json_valid_rate": json_valid_count / total,
            "schema_valid_rate": schema_valid_count / total,
        }

        # Per-task breakdown
        by_task: dict[str, list[EvalResult]] = {}
        for r in results:
            by_task.setdefault(r.task, []).append(r)

        report["by_task"] = {}
        for task, task_results in by_task.items():
            n = len(task_results)
            report["by_task"][task] = {
                "count": n,
                "json_valid_rate": sum(1 for r in task_results if r.json_valid) / n,
                "schema_valid_rate": sum(1 for r in task_results if r.schema_valid) / n,
            }

        return report

    def _evaluate_one(self, pair: SFTPair) -> EvalResult:
        system_msg = pair.messages[0]["content"]
        user_msg = pair.messages[1]["content"]
        ground_truth = pair.messages[2]["content"]

        predicted = self._model_fn(system_msg, user_msg)

        is_json_valid = json_validity(predicted)

        schema = _SCHEMAS.get(pair.task, {})
        is_schema_valid = schema_validity(predicted, schema) if is_json_valid else False

        accuracies: dict[str, float] = {}
        if is_json_valid:
            pred_parsed = parse_model_output(predicted)
            truth_parsed = parse_model_output(ground_truth)
            if pred_parsed is not None and truth_parsed is not None:
                exact = [k for k, v in truth_parsed.items() if isinstance(v, str)]
                numeric = [k for k, v in truth_parsed.items() if isinstance(v, (int, float))]
                accuracies = field_accuracy(
                    pred_parsed, truth_parsed,
                    exact_fields=exact, numeric_fields=numeric,
                )

        return EvalResult(
            stage=pair.source_stage,
            task=pair.task,
            json_valid=is_json_valid,
            schema_valid=is_schema_valid,
            field_accuracy=accuracies,
            semantic_similarity=None,
        )
