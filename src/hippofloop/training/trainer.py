"""Unsloth QLoRA training wrapper.

This module wraps Unsloth + HuggingFace TRL for QLoRA fine-tuning.
The heavy dependencies (torch, unsloth, transformers, trl) are imported
lazily so the rest of the codebase can be used without a GPU.
"""

from __future__ import annotations

import logging
from typing import Any

from hippofloop.protocols import SFTPair, TrainingResult
from hippofloop.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class UnslothTrainer:
    """QLoRA fine-tuning via Unsloth.

    Wraps model loading, LoRA patching, dataset preparation, and training.
    GPU-dependent methods import torch/unsloth/trl lazily.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def prepare_dataset(self, pairs: list[SFTPair]) -> list[dict[str, Any]]:
        """Convert SFTPairs to the dict format expected by SFTTrainer."""
        return [{"messages": pair.messages} for pair in pairs]

    def train(
        self,
        train_data: list[SFTPair],
        val_data: list[SFTPair],
    ) -> TrainingResult:
        """Run QLoRA fine-tuning. Requires GPU.

        Lazy-imports unsloth, transformers, and trl to avoid import
        errors on machines without GPU/CUDA.
        """
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel

        logger.info("Loading base model: %s", self._config.base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._config.base_model,
            max_seq_length=self._config.max_seq_length,
            load_in_4bit=True,
        )

        logger.info(
            "Applying LoRA (rank=%d, alpha=%d)",
            self._config.lora_rank, self._config.lora_alpha,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self._config.lora_rank,
            lora_alpha=self._config.lora_alpha,
            lora_dropout=self._config.lora_dropout,
            target_modules=self._config.lora_target_modules,
        )

        train_dataset = Dataset.from_list(self.prepare_dataset(train_data))
        val_dataset = Dataset.from_list(self.prepare_dataset(val_data))

        output_dir = f"checkpoints/{self._config.base_model.split('/')[-1]}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=self._config.batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            learning_rate=self._config.learning_rate,
            lr_scheduler_type=self._config.lr_scheduler,
            warmup_steps=50,
            weight_decay=self._config.weight_decay,
            bf16=self._config.bf16,
            fp16=self._config.fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=10,
            seed=self._config.seed,
        )

        def formatting_func(example: dict) -> list[str]:
            return [tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
            )]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            formatting_func=formatting_func,
        )

        logger.info("Starting training (%d epochs)", self._config.epochs)
        trainer.train()

        # Find best checkpoint
        best_epoch = 1
        best_loss = float("inf")
        for entry in trainer.state.log_history:
            if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                    best_loss = entry["eval_loss"]
                    best_epoch = int(entry.get("epoch", 1))

        model_path = f"{output_dir}/best"
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        logger.info("Training complete. Best epoch: %d, val_loss: %.4f", best_epoch, best_loss)

        return TrainingResult(
            model_path=model_path,
            val_loss=best_loss,
            epochs_completed=self._config.epochs,
            best_epoch=best_epoch,
        )
