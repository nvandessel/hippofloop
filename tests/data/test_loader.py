# tests/data/test_loader.py
import json

import pytest

from hippofloop.data.loader import JsonlLoader


@pytest.fixture
def decisions_file(tmp_path) -> str:
    """Create a temporary decisions.jsonl with mixed entry types."""
    entries = [
        # Valid summarize entry
        {
            "run_id": "run-001",
            "stage": "extract",
            "pass": "summarize",
            "model": "claude-sonnet-4-6",
            "time": "2026-03-28T00:00:00Z",
            "chunk": 0,
            "prompt": [
                {"role": "system", "content": "You are analyzing..."},
                {"role": "user", "content": "events here"},
            ],
            "response": '{"summary":"test"}',
            "parsed": {"summary": "test"},
        },
        # Bookkeeping entry (no prompt/response)
        {
            "run_id": "run-001",
            "stage": "extract",
            "pass": "start",
            "model": "claude-sonnet-4-6",
            "time": "2026-03-28T00:00:00Z",
            "num_events": 100,
        },
        # Error entry
        {
            "run_id": "run-001",
            "stage": "extract",
            "pass": "summarize",
            "model": "claude-sonnet-4-6",
            "time": "2026-03-28T00:01:00Z",
            "error": "subagent failed",
            "reason": "exit status 1",
            "event": "llm_fallback",
        },
        # Valid extract entry
        {
            "run_id": "run-001",
            "stage": "extract",
            "pass": "extract",
            "model": "claude-sonnet-4-6",
            "time": "2026-03-28T00:02:00Z",
            "chunk": 0,
            "prompt": [
                {"role": "system", "content": "You are extracting..."},
                {"role": "user", "content": "arc + events"},
            ],
            "response": '{"candidates":[]}',
            "parsed": {"candidates": []},
        },
    ]
    path = str(tmp_path / "decisions.jsonl")
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def test_load_parses_all_entries(decisions_file: str):
    loader = JsonlLoader()
    entries = loader.load([decisions_file])
    assert len(entries) == 4


def test_load_maps_pass_field_to_pass_(decisions_file: str):
    """The JSON field 'pass' maps to the Python field 'pass_' (reserved word)."""
    loader = JsonlLoader()
    entries = loader.load([decisions_file])
    assert entries[0].pass_ == "summarize"
    assert entries[1].pass_ == "start"


def test_load_preserves_prompt_structure(decisions_file: str):
    loader = JsonlLoader()
    entries = loader.load([decisions_file])
    assert entries[0].prompt[0]["role"] == "system"
    assert len(entries[0].prompt) == 2


def test_load_handles_missing_optional_fields(decisions_file: str):
    loader = JsonlLoader()
    entries = loader.load([decisions_file])
    # Bookkeeping entry has no prompt, response, parsed, chunk
    bookkeeping = entries[1]
    assert bookkeeping.prompt == []
    assert bookkeeping.response == ""
    assert bookkeeping.parsed is None
    assert bookkeeping.chunk is None


def test_load_detects_error_entries(decisions_file: str):
    loader = JsonlLoader()
    entries = loader.load([decisions_file])
    error_entry = entries[2]
    assert error_entry.error == "subagent failed"


def test_load_multiple_files(tmp_path):
    """Loading from two files concatenates entries."""
    for name in ("a.jsonl", "b.jsonl"):
        with open(str(tmp_path / name), "w") as f:
            f.write(json.dumps({
                "run_id": "run-001", "stage": "extract", "pass": "summarize",
                "model": "test", "time": "2026-03-28T00:00:00Z",
                "prompt": [], "response": "{}", "parsed": {},
            }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])
    assert len(entries) == 2


def test_load_skips_malformed_json_lines(tmp_path):
    """Malformed lines are skipped with a warning, not a crash."""
    path = str(tmp_path / "bad.jsonl")
    with open(path, "w") as f:
        f.write('{"run_id":"run-001","stage":"extract","pass":"summarize","model":"test","time":"T","prompt":[],"response":"{}","parsed":{}}\n')
        f.write("NOT VALID JSON\n")
        f.write('{"run_id":"run-002","stage":"extract","pass":"summarize","model":"test","time":"T","prompt":[],"response":"{}","parsed":{}}\n')
    loader = JsonlLoader()
    entries = loader.load([path])
    assert len(entries) == 2


def test_load_skips_entries_missing_required_fields(tmp_path):
    """Entries missing stage, run_id, or model are skipped."""
    path = str(tmp_path / "incomplete.jsonl")
    with open(path, "w") as f:
        # Missing all required fields
        f.write('{"foo": "bar"}\n')
        # Missing run_id and model
        f.write('{"stage": "extract"}\n')
        # Valid entry
        f.write('{"run_id":"run-001","stage":"extract","pass":"summarize","model":"test","time":"T","prompt":[],"response":"{}","parsed":{}}\n')
    loader = JsonlLoader()
    entries = loader.load([path])
    assert len(entries) == 1


def test_load_fallback_false_not_detected_as_fallback(tmp_path):
    """Entry with {"fallback": false} should have fallback=False."""
    path = str(tmp_path / "fallback.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-001", "stage": "extract", "pass": "summarize",
            "model": "test", "time": "T", "prompt": [], "response": "{}",
            "parsed": {}, "fallback": False,
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert len(entries) == 1
    assert entries[0].fallback is False


def test_load_fallback_true_detected(tmp_path):
    """Entry with {"fallback": true} should have fallback=True."""
    path = str(tmp_path / "fallback_true.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-001", "stage": "extract", "pass": "summarize",
            "model": "test", "time": "T", "prompt": [], "response": "{}",
            "parsed": {}, "fallback": True,
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert entries[0].fallback is True


def test_load_llm_fallback_event_detected(tmp_path):
    """Entry with event=llm_fallback should have fallback=True."""
    path = str(tmp_path / "event_fallback.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-001", "stage": "extract", "pass": "summarize",
            "model": "test", "time": "T", "prompt": [], "response": "{}",
            "parsed": {}, "event": "llm_fallback",
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert entries[0].fallback is True


# -- Sonnet comparison schema fallback --


def test_load_sonnet_comparison_uses_sonnet_response(tmp_path):
    """When 'response' is missing, fall back to 'sonnet_response'."""
    path = str(tmp_path / "sonnet.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-sonnet-001", "stage": "extract", "pass": "arc",
            "model": "sonnet", "time": "2026-03-28T00:00:00Z",
            "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            "sonnet_response": '{"arc":"test arc"}',
            "haiku_response": '{"arc":"haiku arc"}',
            "haiku_parsed": {"arc": "haiku arc"},
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert len(entries) == 1
    assert entries[0].response == '{"arc":"test arc"}'


def test_load_sonnet_comparison_uses_haiku_parsed(tmp_path):
    """When 'parsed' is missing, fall back to 'haiku_parsed'."""
    path = str(tmp_path / "sonnet.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-sonnet-001", "stage": "extract", "pass": "summarize",
            "model": "sonnet", "time": "2026-03-28T00:00:00Z",
            "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            "sonnet_response": '{"summary":"test"}',
            "haiku_parsed": {"summary": "test from haiku"},
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert entries[0].parsed == {"summary": "test from haiku"}


def test_load_prefers_response_over_sonnet_response(tmp_path):
    """When both 'response' and 'sonnet_response' exist, prefer 'response'."""
    path = str(tmp_path / "both.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-001", "stage": "extract", "pass": "summarize",
            "model": "test", "time": "T",
            "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            "response": "original response",
            "parsed": {"original": True},
            "sonnet_response": "sonnet response",
            "haiku_parsed": {"haiku": True},
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert entries[0].response == "original response"
    assert entries[0].parsed == {"original": True}


def test_load_sonnet_comparison_end_to_end(tmp_path):
    """Sonnet comparison entries should survive the full loader pipeline."""
    path = str(tmp_path / "sonnet.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "run_id": "run-sonnet-001", "stage": "extract", "pass": "summarize",
            "model": "sonnet", "time": "2026-03-28T00:00:00Z", "chunk": 0,
            "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            "sonnet_response": '{"summary":"from sonnet"}',
            "haiku_response": '{"summary":"from haiku"}',
            "haiku_parsed": {"summary": "from haiku"},
        }) + "\n")
    loader = JsonlLoader()
    entries = loader.load([path])
    assert len(entries) == 1
    assert entries[0].response == '{"summary":"from sonnet"}'
    assert entries[0].parsed == {"summary": "from haiku"}
    assert entries[0].model == "sonnet"
    assert entries[0].chunk == 0
