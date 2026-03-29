"""Decision entry cleaner — filter, deduplicate, validate."""

from __future__ import annotations

import hashlib
import json
import logging

from hippofloop.protocols import DecisionEntry

logger = logging.getLogger(__name__)

_BOOKKEEPING_PASSES = frozenset({"start", "complete"})


class DecisionCleaner:
    """Filters and deduplicates decision entries for SFT training.

    Drops:
    - Entries missing prompt or response
    - Bookkeeping entries (pass=start/complete)
    - Error entries (error field set)
    - Fallback entries (heuristic-generated, not LLM)
    - Duplicate prompt+response pairs (keeps first occurrence)
    """

    def clean(self, entries: list[DecisionEntry]) -> list[DecisionEntry]:
        result, _ = self.clean_with_stats(entries)
        return result

    def clean_with_stats(
        self, entries: list[DecisionEntry]
    ) -> tuple[list[DecisionEntry], dict[str, int]]:
        stats = {
            "total_input": len(entries),
            "dropped_no_prompt_or_response": 0,
            "dropped_bookkeeping": 0,
            "dropped_error_or_fallback": 0,
            "dropped_duplicate": 0,
            "total_output": 0,
        }

        seen_hashes: set[str] = set()
        cleaned: list[DecisionEntry] = []

        for entry in entries:
            if entry.pass_ in _BOOKKEEPING_PASSES:
                stats["dropped_bookkeeping"] += 1
                continue

            if entry.error is not None or entry.fallback:
                stats["dropped_error_or_fallback"] += 1
                continue

            if not entry.prompt or not entry.response:
                stats["dropped_no_prompt_or_response"] += 1
                continue

            content_hash = self._hash_content(entry)
            if content_hash in seen_hashes:
                stats["dropped_duplicate"] += 1
                continue
            seen_hashes.add(content_hash)

            cleaned.append(entry)

        stats["total_output"] = len(cleaned)
        logger.info(
            "Cleaned %d → %d entries (dropped: %d no-data, %d bookkeeping, "
            "%d error/fallback, %d duplicate)",
            stats["total_input"],
            stats["total_output"],
            stats["dropped_no_prompt_or_response"],
            stats["dropped_bookkeeping"],
            stats["dropped_error_or_fallback"],
            stats["dropped_duplicate"],
        )
        return cleaned, stats

    def _hash_content(self, entry: DecisionEntry) -> str:
        prompt_str = json.dumps(entry.prompt, sort_keys=True)
        content = f"{prompt_str}||{entry.response}"
        return hashlib.sha256(content.encode()).hexdigest()
