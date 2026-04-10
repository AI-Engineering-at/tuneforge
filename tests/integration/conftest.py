"""Pytest configuration and fixtures for integration tests."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="tuneforge_integration_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataset_file(temp_output_dir):
    """Create a sample dataset file for testing."""
    dataset_path = temp_output_dir / "test_dataset.jsonl"
    records = [
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Capital of France?", "input": "", "output": "Paris"},
        {"instruction": "Explain Python", "input": "", "output": "Python is a programming language."},
    ]
    with open(dataset_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return dataset_path


@pytest.fixture
def sample_config(temp_output_dir, sample_dataset_file):
    """Create a sample training configuration."""
    from finetune.trainer import QLoRAConfig

    config = QLoRAConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Small model for testing
        output_dir=str(temp_output_dir / "output"),
        dataset_path=str(sample_dataset_file),
        dataset_format="alpaca",
        backend="peft_trl",
        max_steps=2,  # Minimal steps for testing
        per_device_train_batch_size=1,
        max_seq_length=128,
        lora_r=8,
        lora_alpha=16,
    )
    return config


@pytest.fixture
def mock_hf_datasets():
    """Mock HuggingFace datasets for integration testing."""
    with patch("finetune.trainer.import_hf_datasets_module") as mock_import:
        mock_hf_module = MagicMock()

        # Create a mock Dataset class
        mock_dataset = MagicMock()
        mock_dataset.train_test_split.return_value = {
            "train": MagicMock(),
            "test": MagicMock(),
        }
        mock_hf_module.Dataset.from_list.return_value = mock_dataset

        mock_import.return_value = mock_hf_module
        yield mock_import


@pytest.fixture
def mock_model_loading():
    """Mock model and tokenizer loading for fast tests."""
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
    ):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        yield mock_model, mock_tokenizer


@pytest.fixture
def mock_trl_trainer():
    """Mock TRL SFTTrainer for testing."""
    with patch("trl.SFTTrainer") as mock_trainer_class:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(metrics={"train_loss": 0.5, "eval_loss": 0.4})
        mock_trainer_class.return_value = mock_trainer
        yield mock_trainer_class


@pytest.fixture
def clean_env():
    """Clean environment variables that might affect tests."""
    env_vars_to_clear = [
        "ZEROTH_MOCK_MODE",
        "AEGIS_API_URL",
        "AEGIS_JWT_TOKEN",
        "CUDA_VISIBLE_DEVICES",
    ]
    original_values = {}

    for var in env_vars_to_clear:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
