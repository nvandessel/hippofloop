"""CLI entry points for hippofloop pipeline."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from collections.abc import Callable

from hippofloop.data.cleaner import DecisionCleaner
from hippofloop.data.formatter import SftFormatter
from hippofloop.data.loader import JsonlLoader
from hippofloop.training.config import load_config

logger = logging.getLogger(__name__)


def _cli_error_handler(fn: Callable) -> Callable:
    """Wrap CLI commands to catch common errors and print clean messages."""
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            fn(*args, **kwargs)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="hippofloop — distill floop's consolidator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command")

    # explore command — analyze training data before training
    explore_parser = subparsers.add_parser("explore", help="Analyze decision logs")
    explore_parser.add_argument("paths", nargs="+", help="Paths to decisions.jsonl files")

    # train command
    train_parser = subparsers.add_parser("train", help="Run QLoRA fine-tuning")
    train_parser.add_argument("paths", nargs="+", help="Paths to decisions.jsonl files")
    train_parser.add_argument(
        "--config", default="configs/default.yaml", help="Training config YAML",
    )

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model on test set")
    eval_parser.add_argument("--model", required=True, help="Path to model/adapter")
    eval_parser.add_argument("paths", nargs="+", help="Paths to decisions.jsonl files")
    eval_parser.add_argument(
        "--config", default="configs/default.yaml", help="Training config YAML",
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export to GGUF")
    export_parser.add_argument("--model", required=True, help="Path to trained model")
    export_parser.add_argument("--output", required=True, help="Output GGUF path")
    export_parser.add_argument("--quantization", default="Q4_K_M", help="Quantization method")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "explore":
        _cmd_explore(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "export":
        _cmd_export(args)
    else:
        parser.print_help()
        sys.exit(0)


@_cli_error_handler
def _cmd_explore(args: argparse.Namespace) -> None:
    """Analyze decision logs — counts, token lengths, stage distribution."""
    loader = JsonlLoader()
    entries = loader.load(args.paths)
    print(f"Total entries loaded: {len(entries)}")

    cleaner = DecisionCleaner()
    cleaned, stats = cleaner.clean_with_stats(entries)
    print("\nCleaning stats:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    formatter = SftFormatter()
    pairs = formatter.format(cleaned)
    print("\nSFT pairs by task:")
    by_task: dict[str, int] = {}
    for pair in pairs:
        by_task[pair.task] = by_task.get(pair.task, 0) + 1
    for task, count in sorted(by_task.items()):
        print(f"  {task}: {count}")

    # Token length estimates (rough: 4 chars ≈ 1 token)
    lengths = []
    for pair in pairs:
        total_chars = sum(len(m["content"]) for m in pair.messages)
        lengths.append(total_chars // 4)
    if lengths:
        lengths.sort()
        print("\nEstimated token lengths:")
        print(f"  min: {lengths[0]}")
        print(f"  median: {lengths[len(lengths)//2]}")
        print(f"  p95: {lengths[int(len(lengths)*0.95)]}")
        print(f"  max: {lengths[-1]}")
        over_4096 = sum(1 for tok_len in lengths if tok_len > 4096)
        print(f"  over 4096: {over_4096} ({over_4096/len(lengths)*100:.1f}%)")


@_cli_error_handler
def _cmd_train(args: argparse.Namespace) -> None:
    """Full training pipeline: load → clean → format → split → train."""
    config = load_config(args.config)

    loader = JsonlLoader()
    entries = loader.load(args.paths)

    cleaner = DecisionCleaner()
    cleaned, stats = cleaner.clean_with_stats(entries)
    print(json.dumps(stats, indent=2))

    formatter = SftFormatter()
    pairs = formatter.format(cleaned)

    train, val, test = formatter.split(
        pairs,
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        seed=config.seed,
    )
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    from hippofloop.training.trainer import UnslothTrainer

    trainer = UnslothTrainer(config)
    result = trainer.train(train, val)
    print(f"Training complete: {result}")


def _cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model on test data."""
    print("Eval command requires a trained model. See docs for usage.")
    sys.exit(0)


@_cli_error_handler
def _cmd_export(args: argparse.Namespace) -> None:
    """Export trained model to GGUF."""
    from hippofloop.export.exporter import GgufExporter

    exporter = GgufExporter(quantization=args.quantization)
    output = exporter.export(args.model, args.output)
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()
