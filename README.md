# hippofloop

[![Tests](https://github.com/nvandessel/hippofloop/actions/workflows/test.yml/badge.svg)](https://github.com/nvandessel/hippofloop/actions/workflows/test.yml)
[![CodeQL](https://github.com/nvandessel/hippofloop/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/nvandessel/hippofloop/security/code-scanning)
[![License](https://img.shields.io/github/license/nvandessel/hippofloop)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

> [!WARNING]
> This project is under active development and not yet ready for production use.

Fine-tuning pipeline to distill [floop](https://github.com/nvandessel/floop)'s LLM consolidator into a small local GGUF model.

floop is a spreading-activation memory system for AI coding agents. Its consolidation pipeline uses a cloud LLM, which is slow and costly. hippofloop trains a small local model (3B parameters) to replace it — running entirely offline via [yzma](https://github.com/nvandessel/yzma) GGUF inference.

The name comes from the hippocampus — the brain structure that consolidates short-term to long-term memory during REM sleep.

## Quick Start

```bash
# Install (requires Python 3.11+)
uv sync --extra dev

# Explore training data
hippofloop explore path/to/decisions.jsonl

# Train (requires GPU)
hippofloop train path/to/decisions.jsonl --config configs/default.yaml

# Export to GGUF
hippofloop export --model checkpoints/best --output hippofloop.gguf
```

## Architecture

```
decisions.jsonl → Load → Clean → Format (SFT pairs) → Train (QLoRA) → Export (GGUF)
```

- **Protocol-driven** — all module boundaries are Python Protocols (interfaces)
- **Multi-task model** — single model learns all consolidation stages via task prefixes:
  - `[SUMMARIZE]` — chunk summarization (sub-pass of Extract)
  - `[ARC]` — session arc synthesis (sub-pass of Extract)
  - `[EXTRACT]` — candidate extraction (sub-pass of Extract)
  - `[CLASSIFY]` — memory classification
  - `[RELATE]` — relationship proposals
- **QLoRA via Unsloth** — trains on consumer GPUs (8GB+ VRAM)
- **GGUF export** — deploys as a single file, loaded in-process by yzma

## Development

```bash
uv sync --extra dev
uv run pytest -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
