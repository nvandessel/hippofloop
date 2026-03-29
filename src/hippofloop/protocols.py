"""All interfaces (Protocols) and data types for hippofloop.

This is the single source of truth for all type definitions and interface contracts.
Concrete implementations import from here; they never import each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Task(StrEnum):
    """Canonical task identifiers for the multi-task model.

    Each maps to a consolidation stage/pass in floop's pipeline.
    Used as prefixes in SFT system messages: [SUMMARIZE], [ARC], etc.
    """

    SUMMARIZE = "SUMMARIZE"   # extract.summarize — chunk summarization
    ARC = "ARC"               # extract.arc — session arc synthesis
    EXTRACT = "EXTRACT"       # extract.extract — candidate extraction
    CLASSIFY = "CLASSIFY"     # classify — memory classification
    RELATE = "RELATE"         # relate — relationship proposals


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionEntry:
    """One row from decisions.jsonl.

    Matches the flat JSON schema written by floop's ConsolidationLogger.
    Fields map to the top-level keys in each JSONL line.
    """

    stage: str                              # "extract", "classify", "relate"
    # "summarize", "arc", "extract", or "" for classify/relate
    pass_: str
    prompt: list[dict[str, str]]            # [{"role": "system", "content": "..."}, ...]
    response: str                           # Raw LLM output
    parsed: dict | None                     # Structured JSON the response was parsed into
    run_id: str
    model: str                              # e.g. "claude-sonnet-4-6"
    time: str                               # ISO 8601 timestamp
    # Optional stage-specific fields
    chunk: int | None = None                # Chunk index for extract passes
    error: str | None = None                # Error message if LLM call failed
    fallback: bool = False                  # True if heuristic fallback was used


@dataclass(frozen=True)
class SFTPair:
    """One training example in chat format."""

    messages: list[dict[str, str]]          # system/user/assistant message dicts
    task: str                               # Task prefix: SUMMARIZE, ARC, EXTRACT, CLASSIFY, RELATE
    source_stage: str                       # Original floop stage for traceability


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result for one test example."""

    stage: str
    task: str
    json_valid: bool
    schema_valid: bool
    field_accuracy: dict[str, float]        # field_name → accuracy score
    semantic_similarity: float | None       # Cosine sim, None if not applicable


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training run."""

    model_path: str                         # Path to saved adapter/model
    val_loss: float
    epochs_completed: int
    best_epoch: int


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------

class DataLoader(Protocol):
    """Reads raw decision log files into typed entries."""

    def load(self, paths: Sequence[str]) -> list[DecisionEntry]: ...


class DataCleaner(Protocol):
    """Filters, deduplicates, and validates decision entries."""

    def clean(self, entries: list[DecisionEntry]) -> list[DecisionEntry]: ...

    def clean_with_stats(
        self, entries: list[DecisionEntry],
    ) -> tuple[list[DecisionEntry], dict[str, int]]: ...


class DataFormatter(Protocol):
    """Converts decision entries into SFT training pairs and splits them."""

    def format(self, entries: list[DecisionEntry]) -> list[SFTPair]: ...

    def split(
        self,
        pairs: list[SFTPair],
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> tuple[list[SFTPair], list[SFTPair], list[SFTPair]]: ...


class Trainer(Protocol):
    """Runs QLoRA fine-tuning and returns the result."""

    def train(
        self,
        train_data: list[SFTPair],
        val_data: list[SFTPair],
    ) -> TrainingResult: ...


class Evaluator(Protocol):
    """Runs a model on test data and computes quality metrics."""

    def evaluate(self, test_data: list[SFTPair]) -> list[EvalResult]: ...

    def summary_report(self, results: list[EvalResult]) -> dict[str, Any]: ...


class Exporter(Protocol):
    """Merges LoRA adapter and exports to GGUF."""

    def export(
        self,
        model_path: str,
        output_path: str,
    ) -> str: ...
