# tests/data/test_formatter.py
import pytest

from hippofloop.data.formatter import SftFormatter
from hippofloop.protocols import DecisionEntry, SFTPair


# -- Task prefix mapping --

def test_summarize_entry_gets_summarize_prefix(sample_summarize_entry: DecisionEntry):
    formatter = SftFormatter()
    pairs = formatter.format([sample_summarize_entry])
    assert len(pairs) == 1
    assert pairs[0].task == "SUMMARIZE"
    assert pairs[0].messages[0]["content"].startswith("[SUMMARIZE]")


def test_extract_entry_gets_extract_prefix(sample_extract_entry: DecisionEntry):
    formatter = SftFormatter()
    pairs = formatter.format([sample_extract_entry])
    assert pairs[0].task == "EXTRACT"
    assert pairs[0].messages[0]["content"].startswith("[EXTRACT]")


def test_classify_entry_gets_classify_prefix(sample_classify_entry: DecisionEntry):
    formatter = SftFormatter()
    pairs = formatter.format([sample_classify_entry])
    assert pairs[0].task == "CLASSIFY"
    assert pairs[0].messages[0]["content"].startswith("[CLASSIFY]")


def test_arc_entry_gets_arc_prefix():
    entry = DecisionEntry(
        stage="extract", pass_="arc",
        prompt=[
            {"role": "system", "content": "You are analyzing trajectory."},
            {"role": "user", "content": "summaries here"},
        ],
        response='{"arc":"test"}', parsed={"arc": "test"},
        run_id="r", model="m", time="t",
    )
    formatter = SftFormatter()
    pairs = formatter.format([entry])
    assert pairs[0].task == "ARC"
    assert pairs[0].messages[0]["content"].startswith("[ARC]")


def test_relate_entry_gets_relate_prefix():
    entry = DecisionEntry(
        stage="relate", pass_="",
        prompt=[
            {"role": "system", "content": "You are analyzing relationships."},
            {"role": "user", "content": "memories here"},
        ],
        response='{"relationships":[]}', parsed={"relationships": []},
        run_id="r", model="m", time="t",
    )
    formatter = SftFormatter()
    pairs = formatter.format([entry])
    assert pairs[0].task == "RELATE"


# -- Message structure --

def test_format_produces_three_messages(sample_summarize_entry: DecisionEntry):
    """Each SFT pair has system, user, assistant messages."""
    formatter = SftFormatter()
    pairs = formatter.format([sample_summarize_entry])
    messages = pairs[0].messages
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_assistant_message_is_response(sample_summarize_entry: DecisionEntry):
    formatter = SftFormatter()
    pairs = formatter.format([sample_summarize_entry])
    assert pairs[0].messages[2]["content"] == sample_summarize_entry.response


def test_source_stage_preserved(sample_summarize_entry: DecisionEntry):
    formatter = SftFormatter()
    pairs = formatter.format([sample_summarize_entry])
    assert pairs[0].source_stage == "extract"


# -- Split --

def test_split_ratios():
    formatter = SftFormatter()
    pairs = [
        SFTPair(messages=[], task="SUMMARIZE", source_stage="extract")
        for _ in range(100)
    ]
    train, val, test = formatter.split(pairs, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_is_deterministic():
    formatter = SftFormatter()
    pairs = [
        SFTPair(messages=[{"role": "system", "content": f"[SUMMARIZE] {i}"}], task="SUMMARIZE", source_stage="extract")
        for i in range(50)
    ]
    train1, val1, test1 = formatter.split(pairs, train_ratio=0.8, val_ratio=0.1, seed=42)
    train2, val2, test2 = formatter.split(pairs, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_split_no_overlap():
    formatter = SftFormatter()
    pairs = [
        SFTPair(messages=[{"role": "system", "content": f"[SUMMARIZE] {i}"}], task="SUMMARIZE", source_stage="extract")
        for i in range(50)
    ]
    train, val, test = formatter.split(pairs, train_ratio=0.8, val_ratio=0.1, seed=42)
    all_messages = [p.messages[0]["content"] for p in train + val + test]
    assert len(all_messages) == len(set(all_messages))
