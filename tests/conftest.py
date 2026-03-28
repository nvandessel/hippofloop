"""Shared test fixtures for hippofloop."""

import pytest

from hippofloop.protocols import DecisionEntry, SFTPair


@pytest.fixture
def sample_summarize_entry() -> DecisionEntry:
    """A valid extract.summarize decision entry."""
    return DecisionEntry(
        stage="extract",
        pass_="summarize",
        prompt=[
            {"role": "system", "content": "You are analyzing a chunk of conversation events."},
            {"role": "user", "content": '[evt-1] user: "fix the auth bug"'},
        ],
        response='{"summary":"User asked to fix auth bug","tone":"neutral","phase":"opening","pattern":"debugging","key_moments":[],"open_threads":["auth bug"]}',
        parsed={
            "summary": "User asked to fix auth bug",
            "tone": "neutral",
            "phase": "opening",
            "pattern": "debugging",
            "key_moments": [],
            "open_threads": ["auth bug"],
        },
        run_id="run-test-001",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:00:00Z",
        chunk=0,
    )


@pytest.fixture
def sample_extract_entry() -> DecisionEntry:
    """A valid extract.extract decision entry."""
    return DecisionEntry(
        stage="extract",
        pass_="extract",
        prompt=[
            {"role": "system", "content": "You are extracting behavioral memories."},
            {"role": "user", "content": "Session arc: ... Events: ..."},
        ],
        response='{"candidates":[{"source_events":["evt-42"],"raw_text":"dont mock the db","candidate_type":"correction","confidence":0.92}]}',
        parsed={
            "candidates": [
                {
                    "source_events": ["evt-42"],
                    "raw_text": "dont mock the db",
                    "candidate_type": "correction",
                    "confidence": 0.92,
                }
            ]
        },
        run_id="run-test-001",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:01:00Z",
        chunk=0,
    )


@pytest.fixture
def sample_classify_entry() -> DecisionEntry:
    """A valid classify decision entry."""
    return DecisionEntry(
        stage="classify",
        pass_="",
        prompt=[
            {"role": "system", "content": "You are classifying behavioral memories."},
            {"role": "user", "content": "Candidates: [...]"},
        ],
        response='{"classified":[{"kind":"directive","scope":"universal","importance":0.85}]}',
        parsed={
            "classified": [
                {"kind": "directive", "scope": "universal", "importance": 0.85}
            ]
        },
        run_id="run-test-001",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:02:00Z",
    )


@pytest.fixture
def sample_error_entry() -> DecisionEntry:
    """A decision entry that failed (error)."""
    return DecisionEntry(
        stage="extract",
        pass_="summarize",
        prompt=[{"role": "system", "content": "You are..."}],
        response="",
        parsed=None,
        run_id="run-test-001",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:03:00Z",
        error="subagent failed: exit status 1",
        fallback=True,
    )


@pytest.fixture
def sample_bookkeeping_entry() -> DecisionEntry:
    """A bookkeeping entry (start/complete) — not an LLM call."""
    return DecisionEntry(
        stage="extract",
        pass_="start",
        prompt=[],
        response="",
        parsed=None,
        run_id="run-test-001",
        model="claude-sonnet-4-6",
        time="2026-03-28T00:00:00Z",
    )


@pytest.fixture
def sample_sft_pair() -> SFTPair:
    """A formatted SFT training pair."""
    return SFTPair(
        messages=[
            {"role": "system", "content": "[SUMMARIZE] You are analyzing a chunk of conversation events."},
            {"role": "user", "content": '[evt-1] user: "fix the auth bug"'},
            {"role": "assistant", "content": '{"summary":"User asked to fix auth bug","tone":"neutral","phase":"opening","pattern":"debugging","key_moments":[],"open_threads":["auth bug"]}'},
        ],
        task="SUMMARIZE",
        source_stage="extract",
    )
