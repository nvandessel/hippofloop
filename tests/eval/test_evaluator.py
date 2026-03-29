# tests/eval/test_evaluator.py

import pytest

from hippofloop.eval.evaluator import ModelEvaluator
from hippofloop.protocols import EvalResult, SFTPair


@pytest.fixture
def summarize_test_data() -> list[SFTPair]:
    return [
        SFTPair(
            messages=[
                {"role": "system", "content": "[SUMMARIZE] You are analyzing..."},
                {"role": "user", "content": "events"},
                {
                    "role": "assistant",
                    "content": (
                        '{"summary":"User fixed auth","tone":"satisfied",'
                        '"phase":"resolving","pattern":"debugging",'
                        '"key_moments":[],"open_threads":[]}'
                    ),
                },
            ],
            task="SUMMARIZE",
            source_stage="extract",
        ),
    ]


@pytest.fixture
def mock_model_fn():
    """A mock model function that returns valid JSON."""
    def model_fn(system: str, user: str) -> str:
        return (
            '{"summary":"User fixed auth","tone":"satisfied",'
            '"phase":"resolving","pattern":"debugging",'
            '"key_moments":[],"open_threads":[]}'
        )
    return model_fn


def test_evaluate_returns_eval_results(summarize_test_data: list[SFTPair], mock_model_fn):
    evaluator = ModelEvaluator(model_fn=mock_model_fn)
    results = evaluator.evaluate(test_data=summarize_test_data)
    assert len(results) == 1
    assert isinstance(results[0], EvalResult)


def test_evaluate_json_valid(summarize_test_data: list[SFTPair], mock_model_fn):
    evaluator = ModelEvaluator(model_fn=mock_model_fn)
    results = evaluator.evaluate(test_data=summarize_test_data)
    assert results[0].json_valid is True


def test_evaluate_task_preserved(summarize_test_data: list[SFTPair], mock_model_fn):
    evaluator = ModelEvaluator(model_fn=mock_model_fn)
    results = evaluator.evaluate(test_data=summarize_test_data)
    assert results[0].task == "SUMMARIZE"


def test_evaluate_invalid_json():
    test_data = [
        SFTPair(
            messages=[
                {"role": "system", "content": "[SUMMARIZE] You are..."},
                {"role": "user", "content": "events"},
                {"role": "assistant", "content": '{"valid": true}'},
            ],
            task="SUMMARIZE",
            source_stage="extract",
        ),
    ]

    def bad_model_fn(system: str, user: str) -> str:
        return "NOT VALID JSON AT ALL"

    evaluator = ModelEvaluator(model_fn=bad_model_fn)
    results = evaluator.evaluate(test_data=test_data)
    assert results[0].json_valid is False


def test_evaluate_summary_report(summarize_test_data: list[SFTPair], mock_model_fn):
    evaluator = ModelEvaluator(model_fn=mock_model_fn)
    results = evaluator.evaluate(test_data=summarize_test_data)
    report = evaluator.summary_report(results)
    assert "json_valid_rate" in report
    assert report["json_valid_rate"] == 1.0
    assert report["total"] == 1
