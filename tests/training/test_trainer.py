# tests/training/test_trainer.py

import pytest

from hippofloop.protocols import SFTPair
from hippofloop.training.config import TrainingConfig
from hippofloop.training.trainer import UnslothTrainer


@pytest.fixture
def mock_config() -> TrainingConfig:
    return TrainingConfig(
        base_model="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        lora_rank=32, lora_alpha=64, lora_dropout=0.05,
        lora_target_modules=["q_proj"],
        learning_rate=2e-4, lr_scheduler="cosine", warmup_ratio=0.03,
        epochs=3, batch_size=4, gradient_accumulation_steps=4,
        max_seq_length=4096, weight_decay=0.01, bf16=True, fp16=False,
        train_split=0.8, val_split=0.1, test_split=0.1, seed=42,
        quantization="Q4_K_M", output_path="/tmp/hippofloop.gguf",
    )


@pytest.fixture
def train_data() -> list[SFTPair]:
    return [
        SFTPair(
            messages=[
                {"role": "system", "content": "[SUMMARIZE] You are..."},
                {"role": "user", "content": "events"},
                {"role": "assistant", "content": '{"summary":"test"}'},
            ],
            task="SUMMARIZE",
            source_stage="extract",
        )
        for _ in range(10)
    ]


def test_prepare_dataset_converts_sft_pairs(mock_config: TrainingConfig, train_data: list[SFTPair]):
    trainer = UnslothTrainer(mock_config)
    dataset = trainer.prepare_dataset(train_data)
    assert len(dataset) == 10
    assert "messages" in dataset[0]


def test_prepare_dataset_preserves_message_structure(
    mock_config: TrainingConfig, train_data: list[SFTPair],
):
    trainer = UnslothTrainer(mock_config)
    dataset = trainer.prepare_dataset(train_data)
    messages = dataset[0]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[2]["role"] == "assistant"
