# tests/training/test_config.py
import pytest
import yaml

from hippofloop.training.config import TrainingConfig, load_config


@pytest.fixture
def config_file(tmp_path) -> str:
    config = {
        "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        },
        "training": {
            "learning_rate": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "weight_decay": 0.01,
            "bf16": True,
            "fp16": False,
        },
        "data": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,
        },
        "export": {
            "quantization": "Q4_K_M",
            "output_path": "~/.floop/models/hippofloop.gguf",
        },
    }
    path = str(tmp_path / "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_load_config_from_yaml(config_file: str):
    config = load_config(config_file)
    assert isinstance(config, TrainingConfig)
    assert config.base_model == "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def test_lora_fields(config_file: str):
    config = load_config(config_file)
    assert config.lora_rank == 32
    assert config.lora_alpha == 64
    assert config.lora_dropout == 0.05
    assert "q_proj" in config.lora_target_modules


def test_training_fields(config_file: str):
    config = load_config(config_file)
    assert config.learning_rate == 2e-4
    assert config.epochs == 3
    assert config.batch_size == 4
    assert config.max_seq_length == 4096
    assert config.bf16 is True
    assert config.fp16 is False


def test_data_fields(config_file: str):
    config = load_config(config_file)
    assert config.train_split == 0.8
    assert config.val_split == 0.1
    assert config.seed == 42


def test_export_fields(config_file: str):
    config = load_config(config_file)
    assert config.quantization == "Q4_K_M"
    assert "hippofloop.gguf" in config.output_path


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_splits_sum_to_one(config_file: str):
    config = load_config(config_file)
    total = config.train_split + config.val_split + config.test_split
    assert abs(total - 1.0) < 1e-9
