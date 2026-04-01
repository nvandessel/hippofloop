"""JSONL decision log loader."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from hippofloop.protocols import DecisionEntry

logger = logging.getLogger(__name__)


class JsonlLoader:
    """Reads decisions.jsonl files into DecisionEntry objects.

    Handles the mapping from JSON field names to Python dataclass fields,
    including the 'pass' → 'pass_' rename (pass is a reserved word).
    """

    def load(self, paths: Sequence[str]) -> list[DecisionEntry]:
        entries: list[DecisionEntry] = []
        for path in paths:
            entries.extend(self._load_file(path))
        return entries

    _REQUIRED_FIELDS = ("stage", "run_id", "model")

    def _load_file(self, path: str) -> list[DecisionEntry]:
        entries: list[DecisionEntry] = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at %s:%d", path, line_num)
                    continue
                missing = [f for f in self._REQUIRED_FIELDS if not raw.get(f)]
                if missing:
                    logger.warning(
                        "Skipping entry missing required fields %s at %s:%d",
                        missing, path, line_num,
                    )
                    continue
                entries.append(self._parse_entry(raw))
        return entries

    def _parse_entry(self, raw: dict) -> DecisionEntry:
        return DecisionEntry(
            stage=raw.get("stage", ""),
            pass_=raw.get("pass", ""),
            prompt=raw.get("prompt", []),
            response=(
                raw.get("response")
                if raw.get("response") is not None
                else raw.get("sonnet_response", "")
            ),
            parsed=(
                raw.get("parsed")
                if raw.get("parsed") is not None
                else raw.get("haiku_parsed")
            ),
            run_id=raw.get("run_id", ""),
            model=raw.get("model", ""),
            time=raw.get("time", raw.get("timestamp", "")),
            chunk=raw.get("chunk"),
            error=raw.get("error"),
            fallback=bool(raw.get("fallback", False)) or raw.get("event") == "llm_fallback",
        )
