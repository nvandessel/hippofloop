"""Training configuration — load from YAML, expose as typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """All hyperparameters and paths for a training run.

    Loaded from a YAML file. No defaults — everything must be explicit
    in the config file for reproducibility.
    """

    # Model
    base_model: str

    # LoRA
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]

    # Training
    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    weight_decay: float
    bf16: bool

    # Data splits
    train_split: float
    val_split: float
    test_split: float
    seed: int

    # Export
    quantization: str
    output_path: str


def load_config(path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    Raises FileNotFoundError if the file doesn't exist.
    Raises KeyError if required fields are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    lora = raw["lora"]
    training = raw["training"]
    data = raw["data"]
    export = raw["export"]

    return TrainingConfig(
        base_model=raw["base_model"],
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        lora_target_modules=lora["target_modules"],
        learning_rate=training["learning_rate"],
        lr_scheduler=training["lr_scheduler"],
        warmup_ratio=training["warmup_ratio"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        max_seq_length=training["max_seq_length"],
        weight_decay=training["weight_decay"],
        bf16=training["bf16"],
        train_split=data["train_split"],
        val_split=data["val_split"],
        test_split=data["test_split"],
        seed=data["seed"],
        quantization=export["quantization"],
        output_path=export["output_path"],
    )
