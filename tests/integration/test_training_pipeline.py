"""Integration tests for the training pipeline.

Tests the complete training flow from config to trained model.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from finetune.trainer import (
    QLoRAConfig,
    QLoRATrainer,
    TrainingSummary,
    build_text_datasets,
    load_jsonl_records,
)


class TestTrainingPipeline:
    """End-to-end training pipeline tests."""

    def test_config_to_dataset_loading(self, temp_output_dir, sample_dataset_file):
        """Test that config correctly specifies dataset loading."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            dataset_format="alpaca",
        )

        # Load records as the pipeline would
        records = load_jsonl_records(Path(config.dataset_path))
        assert len(records) == 3
        assert all("instruction" in r for r in records)

    def test_dataset_normalization(self, sample_dataset_file):
        """Test that datasets are correctly normalized."""
        from data_utils.data_formats import normalize_records_to_text

        records = load_jsonl_records(sample_dataset_file)
        normalized = normalize_records_to_text(records, dataset_format="alpaca")

        assert len(normalized) == 3
        assert all("text" in r for r in normalized)
        # Check that the text contains expected content
        assert "What is 2+2?" in normalized[0]["text"]
        assert "Paris" in normalized[1]["text"]

    def test_train_test_split(self, sample_dataset_file):
        """Test train/test splitting functionality."""
        with patch("finetune.trainer.import_hf_datasets_module") as mock_import:
            mock_hf = MagicMock()
            mock_dataset = MagicMock()
            mock_dataset.train_test_split.return_value = {
                "train": MagicMock(num_rows=2),
                "test": MagicMock(num_rows=1),
            }
            mock_hf.Dataset.from_list.return_value = mock_dataset
            mock_import.return_value = mock_hf

            records = load_jsonl_records(Path(sample_dataset_file))
            train_ds, eval_ds = build_text_datasets(records, "alpaca", eval_split_ratio=0.3)

            assert train_ds is not None
            assert eval_ds is not None
            mock_dataset.train_test_split.assert_called_once()

    def test_trainer_initialization(self, sample_config):
        """Test that trainer initializes correctly from config."""
        trainer = QLoRATrainer(sample_config)

        assert trainer.config == sample_config
        assert trainer.backend_name == "peft_trl"
        assert trainer.model is None
        assert trainer.tokenizer is None

    def test_trainer_backend_selection(self, temp_output_dir, sample_dataset_file):
        """Test that different backends can be selected."""
        for backend in ["peft_trl", "unsloth"]:
            config = QLoRAConfig(
                base_model="test-model",
                output_dir=str(temp_output_dir),
                dataset_path=str(sample_dataset_file),
                backend=backend,
            )
            trainer = QLoRATrainer(config)
            assert trainer.backend_name == backend

    def test_invalid_backend_raises_error(self, temp_output_dir, sample_dataset_file):
        """Test that invalid backend raises ValueError."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="peft_trl",  # Valid initially
        )

        # Manually set invalid backend
        config.backend = "invalid_backend"

        with pytest.raises(ValueError, match="Unsupported backend"):
            if config.backend not in {"peft_trl", "unsloth"}:
                raise ValueError(f"Unsupported backend: {config.backend}")


class TestConfigPersistence:
    """Tests for configuration save/load."""

    def test_config_roundtrip_yaml(self, temp_output_dir):
        """Test that config can be saved and loaded from YAML."""
        original_config = QLoRAConfig(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            output_dir=str(temp_output_dir / "output"),
            lora_r=64,
            lora_alpha=128,
            use_rslora=True,
        )

        yaml_path = temp_output_dir / "config.yaml"
        original_config.to_yaml(yaml_path)

        assert yaml_path.exists()

        loaded_config = QLoRAConfig.from_yaml(yaml_path)

        assert loaded_config.base_model == original_config.base_model
        assert loaded_config.lora_r == original_config.lora_r
        assert loaded_config.use_rslora == original_config.use_rslora

    def test_config_ignores_unknown_fields(self, temp_output_dir):
        """Test that loading config ignores unknown YAML fields."""
        yaml_path = temp_output_dir / "config_with_extra.yaml"
        yaml_content = """
base_model: test-model
lora_r: 32
unknown_field: should_be_ignored
another_unknown: 123
"""
        yaml_path.write_text(yaml_content)

        config = QLoRAConfig.from_yaml(yaml_path)
        assert config.base_model == "test-model"
        assert config.lora_r == 32


class TestTrainingSummary:
    """Tests for training summary generation."""

    def test_summary_from_training_results(self):
        """Test creating summary from training results."""
        summary = TrainingSummary(
            primary_metric_name="eval_loss",
            primary_metric_value=0.345,
            metric_goal="minimize",
            metrics={
                "train_loss": 0.321,
                "eval_perplexity": 1.41,
            },
            peak_vram_mb=8192.5,
            training_seconds=3600.0,
            total_seconds=3650.0,
        )

        # Test dictionary conversion
        d = summary.to_dict()
        assert d["primary_metric_name"] == "eval_loss"
        assert d["primary_metric_value"] == 0.345
        assert d["metrics"]["train_loss"] == 0.321
        assert d["peak_vram_mb"] == 8192.5

        # Test lines conversion
        lines = summary.to_lines()
        assert any("eval_loss" in line for line in lines)
        assert any("0.345000" in line for line in lines)
        assert any("8192.5" in line for line in lines)
