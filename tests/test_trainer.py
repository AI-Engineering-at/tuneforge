"""Tests for the hybrid QLoRA trainer and CLI."""
import os
import sys
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import finetune.trainer as trainer_mod
from finetune.trainer import (
    QLoRAConfig,
    TrainingSummary,
    build_text_datasets,
    load_jsonl_records,
    main,
    run_from_config,
)


def test_qlora_config_defaults():
    config = QLoRAConfig(base_model="Qwen/Qwen2.5-Coder-7B-Instruct")
    assert config.backend == "peft_trl"
    assert config.dataset_format == "alpaca"
    assert config.primary_metric == "eval_loss"
    assert config.metric_goal == "minimize"
    assert config.lora_dropout == 0.0
    assert config.use_rslora is False


def test_qlora_config_from_yaml(tmp_path):
    yaml_content = """
base_model: "Qwen/Qwen3-8B"
backend: "unsloth"
dataset_path: "datasets/generated/sps"
dataset_format: "alpaca"
primary_metric: "eval_loss"
metric_goal: "minimize"
lora_r: 32
lora_alpha: 64
learning_rate: 1e-4
max_steps: 2000
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = QLoRAConfig.from_yaml(config_file)
    assert config.base_model == "Qwen/Qwen3-8B"
    assert config.backend == "unsloth"
    assert config.dataset_path == "datasets/generated/sps"
    assert config.lora_r == 32
    assert config.max_steps == 2000


def test_qlora_config_to_yaml(tmp_path):
    config = QLoRAConfig(base_model="test-model", lora_r=64, backend="unsloth")
    out_path = tmp_path / "out.yaml"
    config.to_yaml(out_path)
    loaded = QLoRAConfig.from_yaml(out_path)
    assert loaded.lora_r == 64
    assert loaded.base_model == "test-model"
    assert loaded.backend == "unsloth"


def test_qlora_config_ignores_unknown_yaml_fields(tmp_path):
    yaml_content = """
base_model: "test"
lora_r: 8
unknown_field: "should be ignored"
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = QLoRAConfig.from_yaml(config_file)
    assert config.lora_r == 8


def test_load_jsonl_records_from_file(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction":"Q","input":"","output":"A"}\n')
    records = load_jsonl_records(dataset)
    assert len(records) == 1
    assert records[0]["instruction"] == "Q"


def test_load_jsonl_records_from_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "part1.jsonl").write_text('{"text":"one"}\n')
    (data_dir / "part2.jsonl").write_text('{"text":"two"}\n')
    records = load_jsonl_records(data_dir)
    assert [record["text"] for record in records] == ["one", "two"]


def test_build_text_datasets_split():
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = ["text"]

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def train_test_split(self, test_size, seed):
            split_idx = max(1, len(self.rows) - 1)
            return {
                "train": FakeDataset(self.rows[:split_idx]),
                "test": FakeDataset(self.rows[split_idx:]),
            }

    class FakeDatasetsModule:
        Dataset = FakeDataset

    with patch.object(
        trainer_mod,
        "import_hf_datasets_module",
        return_value=FakeDatasetsModule(),
    ):
        train_dataset, eval_dataset = build_text_datasets([
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "", "output": "A2"},
            {"instruction": "Q3", "input": "", "output": "A3"},
        ], dataset_format="alpaca", eval_split_ratio=0.34)
    assert train_dataset is not None
    assert eval_dataset is not None
    assert "text" in train_dataset.column_names


def test_training_summary_to_lines():
    summary = TrainingSummary(
        primary_metric_name="eval_loss",
        primary_metric_value=1.23,
        metric_goal="minimize",
        metrics={"eval_loss": 1.23, "train_loss": 1.11},
        peak_vram_mb=4096,
        training_seconds=10,
        total_seconds=12,
    )
    lines = summary.to_lines()
    assert "primary_metric_name: eval_loss" in lines
    assert any(line.startswith("peak_vram_mb: ") for line in lines)


def test_cli_main_smoke(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset_path: 'datasets/generated/sps'\n")
    summary_path = tmp_path / "summary.json"

    def fake_run_from_config(config_path, backend=None, dataset_path=None, do_eval=False):
        assert backend == "unsloth"
        assert dataset_path == "tmp-dataset"
        assert do_eval is True
        return TrainingSummary(
            primary_metric_name="eval_loss",
            primary_metric_value=0.75,
            metric_goal="minimize",
            metrics={"eval_loss": 0.75},
            peak_vram_mb=8192,
            training_seconds=30,
            total_seconds=33,
        )

    monkeypatch.setattr("finetune.trainer.run_from_config", fake_run_from_config)
    rc = main([
        "--config", str(config_path),
        "--backend", "unsloth",
        "--dataset", "tmp-dataset",
        "--eval",
        "--summary-json", str(summary_path),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "primary_metric_name: eval_loss" in captured.out
    assert "primary_metric_value: 0.750000" in captured.out
    assert summary_path.exists()


def test_run_from_config_uses_trainer(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
base_model: "Qwen/Qwen3-8B"
dataset_path: "ignored"
primary_metric: "eval_loss"
metric_goal: "minimize"
output_dir: "output/test"
"""
    )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.saved_to = None

        def setup(self):
            return None

        def train(self, train_dataset, eval_dataset=None):
            assert train_dataset == "train-dataset"
            assert eval_dataset == "eval-dataset"
            return object(), TrainingSummary(
                primary_metric_name="eval_loss",
                primary_metric_value=0.9,
                metric_goal="minimize",
                metrics={"eval_loss": 0.9},
                peak_vram_mb=2048,
                training_seconds=5,
                total_seconds=6,
            )

        def save(self, path):
            self.saved_to = path

    monkeypatch.setattr("finetune.trainer.load_jsonl_records", lambda _: [{"text": "x"}, {"text": "y"}])
    monkeypatch.setattr(
        "finetune.trainer.build_text_datasets",
        lambda records, dataset_format, eval_split_ratio: ("train-dataset", "eval-dataset"),
    )
    monkeypatch.setattr("finetune.trainer.QLoRATrainer", FakeTrainer)

    summary = run_from_config(config_path, backend="unsloth", dataset_path="override", do_eval=True)
    assert summary.primary_metric_name == "eval_loss"
    assert summary.primary_metric_value == 0.9


def test_control_configs_load():
    root = Path(__file__).resolve().parents[1] / "finetune" / "configs"
    sps = QLoRAConfig.from_yaml(root / "sps-plc.yaml")
    legal = QLoRAConfig.from_yaml(root / "legal-dsgvo.yaml")
    assert sps.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert legal.base_model == "LeoLM/leo-mistral-hessianai-7b"
    assert sps.lora_dropout == 0.0
    assert legal.lora_dropout == 0.0


def test_qwen3_candidate_configs_load():
    root = Path(__file__).resolve().parents[1] / "finetune" / "configs"
    sps = QLoRAConfig.from_yaml(root / "sps-plc-qwen3.yaml")
    legal = QLoRAConfig.from_yaml(root / "legal-dsgvo-qwen3.yaml")
    assert sps.base_model == "Qwen/Qwen3-8B"
    assert legal.base_model == "Qwen/Qwen3-8B"
    assert sps.primary_metric == "eval_loss"
    assert legal.primary_metric == "eval_loss"
