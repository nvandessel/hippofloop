from hippofloop.protocols import DecisionEntry, EvalResult, SFTPair, TrainingResult


def test_decision_entry_construction():
    entry = DecisionEntry(
        stage="extract",
        pass_="summarize",
        prompt=[{"role": "system", "content": "You are..."}],
        response='{"summary": "test"}',
        parsed={"summary": "test"},
        run_id="run-123",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:00:00Z",
    )
    assert entry.stage == "extract"
    assert entry.pass_ == "summarize"
    assert entry.chunk is None
    assert entry.error is None
    assert entry.fallback is False


def test_decision_entry_with_optional_fields():
    entry = DecisionEntry(
        stage="extract",
        pass_="summarize",
        prompt=[],
        response="{}",
        parsed={},
        run_id="run-123",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:00:00Z",
        chunk=3,
        error="timeout",
        fallback=True,
    )
    assert entry.chunk == 3
    assert entry.error == "timeout"
    assert entry.fallback is True


def test_sft_pair_construction():
    pair = SFTPair(
        messages=[
            {"role": "system", "content": "[SUMMARIZE] You are..."},
            {"role": "user", "content": "events"},
            {"role": "assistant", "content": "summary"},
        ],
        task="SUMMARIZE",
        source_stage="extract",
    )
    assert pair.task == "SUMMARIZE"
    assert len(pair.messages) == 3


def test_eval_result_construction():
    result = EvalResult(
        stage="extract",
        task="SUMMARIZE",
        json_valid=True,
        schema_valid=True,
        field_accuracy={"tone": 1.0, "phase": 0.8},
        semantic_similarity=0.92,
    )
    assert result.json_valid is True
    assert result.semantic_similarity == 0.92


def test_eval_result_without_semantic_similarity():
    result = EvalResult(
        stage="classify",
        task="CLASSIFY",
        json_valid=True,
        schema_valid=True,
        field_accuracy={"kind": 0.9},
        semantic_similarity=None,
    )
    assert result.semantic_similarity is None


def test_training_result_construction():
    result = TrainingResult(
        model_path="/tmp/model",
        val_loss=0.45,
        epochs_completed=3,
        best_epoch=2,
    )
    assert result.val_loss == 0.45
