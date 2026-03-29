# tests/data/test_cleaner.py
import pytest

from hippofloop.data.cleaner import DecisionCleaner
from hippofloop.protocols import DecisionEntry


def test_drops_entries_without_prompt(sample_error_entry: DecisionEntry):
    """Entries with empty prompt are dropped."""
    cleaner = DecisionCleaner()
    # sample_error_entry has a prompt but empty response — make one with no prompt
    no_prompt = DecisionEntry(
        stage="extract", pass_="summarize", prompt=[], response="something",
        parsed=None, run_id="r", model="m", time="t",
    )
    result = cleaner.clean([no_prompt])
    assert len(result) == 0


def test_drops_entries_without_response(sample_summarize_entry: DecisionEntry):
    """Entries with empty response are dropped."""
    cleaner = DecisionCleaner()
    no_response = DecisionEntry(
        stage="extract", pass_="summarize",
        prompt=[{"role": "system", "content": "x"}],
        response="", parsed=None, run_id="r", model="m", time="t",
    )
    result = cleaner.clean([no_response])
    assert len(result) == 0


def test_drops_bookkeeping_entries(sample_bookkeeping_entry: DecisionEntry):
    """Entries with pass 'start' or 'complete' are dropped."""
    cleaner = DecisionCleaner()
    result = cleaner.clean([sample_bookkeeping_entry])
    assert len(result) == 0


def test_drops_error_entries(sample_error_entry: DecisionEntry):
    """Entries with error field set are dropped."""
    cleaner = DecisionCleaner()
    result = cleaner.clean([sample_error_entry])
    assert len(result) == 0


def test_drops_fallback_entries():
    """Entries where fallback=True are dropped."""
    cleaner = DecisionCleaner()
    entry = DecisionEntry(
        stage="extract", pass_="summarize",
        prompt=[{"role": "system", "content": "x"}],
        response='{"summary":"test"}', parsed={"summary": "test"},
        run_id="r", model="m", time="t", fallback=True,
    )
    result = cleaner.clean([entry])
    assert len(result) == 0


def test_keeps_valid_entries(
    sample_summarize_entry: DecisionEntry,
    sample_extract_entry: DecisionEntry,
    sample_classify_entry: DecisionEntry,
):
    cleaner = DecisionCleaner()
    result = cleaner.clean([
        sample_summarize_entry,
        sample_extract_entry,
        sample_classify_entry,
    ])
    assert len(result) == 3


def test_deduplicates_by_content_hash(sample_summarize_entry: DecisionEntry):
    """Identical prompt+response pairs are deduplicated."""
    cleaner = DecisionCleaner()
    result = cleaner.clean([sample_summarize_entry, sample_summarize_entry])
    assert len(result) == 1


def test_dedup_keeps_first_occurrence(sample_summarize_entry: DecisionEntry):
    """When deduplicating, the first occurrence is kept."""
    later = DecisionEntry(
        stage=sample_summarize_entry.stage,
        pass_=sample_summarize_entry.pass_,
        prompt=sample_summarize_entry.prompt,
        response=sample_summarize_entry.response,
        parsed=sample_summarize_entry.parsed,
        run_id="run-different",
        model=sample_summarize_entry.model,
        time="2026-03-29T00:00:00Z",
    )
    cleaner = DecisionCleaner()
    result = cleaner.clean([sample_summarize_entry, later])
    assert len(result) == 1
    assert result[0].run_id == sample_summarize_entry.run_id


def test_clean_returns_stats(
    sample_summarize_entry: DecisionEntry,
    sample_error_entry: DecisionEntry,
    sample_bookkeeping_entry: DecisionEntry,
):
    """clean_with_stats returns both cleaned entries and stats dict."""
    cleaner = DecisionCleaner()
    result, stats = cleaner.clean_with_stats([
        sample_summarize_entry,
        sample_error_entry,
        sample_bookkeeping_entry,
    ])
    assert len(result) == 1
    assert stats["total_input"] == 3
    assert stats["dropped_no_prompt_or_response"] >= 0
    assert stats["dropped_bookkeeping"] >= 1
    assert stats["dropped_error_or_fallback"] >= 1
    assert stats["dropped_duplicate"] >= 0
    assert stats["total_output"] == 1
