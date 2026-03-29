# src/hippofloop/data/formatter.py
"""Convert decision entries into SFT chat-format training pairs."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

from hippofloop.protocols import DecisionEntry, SFTPair

logger = logging.getLogger(__name__)

# Maps (stage, pass_) to task prefix
_TASK_MAP: dict[tuple[str, str], str] = {
    ("extract", "summarize"): "SUMMARIZE",
    ("extract", "arc"): "ARC",
    ("extract", "extract"): "EXTRACT",
    ("classify", ""): "CLASSIFY",
    ("relate", ""): "RELATE",
}


class SftFormatter:
    """Converts cleaned DecisionEntry objects into SFT training pairs.

    Each pair has three messages (system/user/assistant) with a task prefix
    prepended to the system message. The task prefix tells the multi-task
    model which consolidation operation to perform.
    """

    def format(self, entries: Sequence[DecisionEntry]) -> list[SFTPair]:
        pairs: list[SFTPair] = []
        for entry in entries:
            task = self._resolve_task(entry)
            if task is None:
                continue
            pair = self._format_entry(entry, task)
            pairs.append(pair)
        return pairs

    def split(
        self,
        pairs: list[SFTPair],
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> tuple[list[SFTPair], list[SFTPair], list[SFTPair]]:
        rng = random.Random(seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = round(n * train_ratio)
        val_end = train_end + round(n * val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        if n >= 3:
            assert train, "Train split is empty"
            assert val, "Validation split is empty"
            assert test, "Test split is empty"

        return train, val, test

    def _resolve_task(self, entry: DecisionEntry) -> str | None:
        key = (entry.stage, entry.pass_)
        task = _TASK_MAP.get(key)
        if task is None:
            # Try with empty pass_ for stages that don't use sub-passes
            task = _TASK_MAP.get((entry.stage, ""))
        return task

    def _format_entry(self, entry: DecisionEntry, task: str) -> SFTPair:
        # Build system message: prepend task prefix to original system content
        system_content = ""
        user_content = ""
        system_count = 0
        user_count = 0
        for msg in entry.prompt:
            if msg["role"] == "system":
                system_content = msg["content"]
                system_count += 1
            elif msg["role"] == "user":
                user_content = msg["content"]
                user_count += 1
        if system_count > 1:
            logger.warning(
                "Entry %s/%s has %d system messages; using last",
                entry.stage, entry.pass_, system_count,
            )
        if user_count > 1:
            logger.warning(
                "Entry %s/%s has %d user messages; using last",
                entry.stage, entry.pass_, user_count,
            )

        messages = [
            {"role": "system", "content": f"[{task}] {system_content}"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": entry.response},
        ]

        return SFTPair(
            messages=messages,
            task=task,
            source_stage=entry.stage,
        )
