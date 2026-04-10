"""Hybrid QLoRA fine-tuning runtime for domain-specific models.

Supports a stable `transformers + peft + trl` backend plus an optional
`unsloth + trl` backend behind config/CLI selection.

Usage:
    python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from data_utils.data_formats import (
    SUPPORTED_DATASET_FORMATS,
    normalize_records_to_text,
)

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = {"peft_trl", "unsloth"}
SUPPORTED_METRIC_GOALS = {"maximize", "minimize"}


@dataclass
class TrainingSummary:
    primary_metric_name: str
    primary_metric_value: float
    metric_goal: str
    metrics: dict[str, float] = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "metric_goal": self.metric_goal,
            "metrics": self.metrics,
            "peak_vram_mb": self.peak_vram_mb,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
        }

    def to_lines(self) -> list[str]:
        lines = [
            f"primary_metric_name: {self.primary_metric_name}",
            f"primary_metric_value: {self.primary_metric_value:.6f}",
            f"metric_goal: {self.metric_goal}",
        ]
        for key in sorted(self.metrics):
            lines.append(f"{key}: {self.metrics[key]:.6f}")
        lines.extend(
            [
                f"peak_vram_mb: {self.peak_vram_mb:.1f}",
                f"training_seconds: {self.training_seconds:.2f}",
                f"total_seconds: {self.total_seconds:.2f}",
            ]
        )
        return lines


@dataclass
class QLoRAConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "output"
    backend: str = "peft_trl"

    # Dataset
    dataset_path: str = ""
    dataset_format: str = "alpaca"
    eval_split_ratio: float = 0.1

    # Metrics
    primary_metric: str = "eval_loss"
    metric_goal: str = "minimize"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    use_rslora: bool = False
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Quantization
    bits: int = 4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    learning_rate: float = 2e-4
    max_steps: int = 1000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    def __post_init__(self):
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.dataset_format not in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f"Unsupported dataset_format: {self.dataset_format}")
        if self.metric_goal not in SUPPORTED_METRIC_GOALS:
            raise ValueError(f"Unsupported metric_goal: {self.metric_goal}")

    @classmethod
    def from_yaml(cls, path: Path) -> "QLoRAConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        filtered = {}
        for key, value in data.items():
            if key not in cls.__dataclass_fields__:
                continue
            field_type = cls.__dataclass_fields__[key].type
            if field_type is float and isinstance(value, str):
                value = float(value)
            elif field_type is int and isinstance(value, str):
                value = int(value)
            filtered[key] = value
        return cls(**filtered)

    def to_yaml(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def load_jsonl_records(path: Path) -> list[dict]:
    """Load training records from a JSONL file or directory of JSONL files."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in dataset path: {path}")

    records: list[dict] = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path} line {line_number}") from exc
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def import_hf_datasets_module():
    """Import HuggingFace `datasets` without colliding with the local `datasets/` package."""
    project_root = Path(__file__).resolve().parents[1]
    saved_modules = {
        name: module for name, module in list(sys.modules.items()) if name == "datasets" or name.startswith("datasets.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    original_path = list(sys.path)
    try:
        sys.path = [path for path in sys.path if Path(path).resolve() != project_root]
        return importlib.import_module("datasets")
    finally:
        imported_modules = [name for name in list(sys.modules) if name == "datasets" or name.startswith("datasets.")]
        for name in imported_modules:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path = original_path


def build_text_datasets(records: list[dict], dataset_format: str, eval_split_ratio: float):
    """Normalize records to text and split into train/eval datasets."""
    hf_datasets = import_hf_datasets_module()
    Dataset = hf_datasets.Dataset
    normalized = normalize_records_to_text(records, dataset_format=dataset_format)
    dataset = Dataset.from_list(normalized)

    if len(normalized) < 2 or eval_split_ratio <= 0:
        return dataset, None

    split = dataset.train_test_split(test_size=eval_split_ratio, seed=42)
    return split["train"], split["test"]


class QLoRATrainer:
    """QLoRA training wrapper using PEFT/TRL or optional Unsloth/TRL."""

    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.backend_name = config.backend

    def setup(self):
        """Load model with 4-bit quantization and apply LoRA."""
        if self.config.backend == "unsloth":
            self._setup_unsloth()
            return
        self._setup_peft_trl()

    def _setup_peft_trl(self):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=self.config.use_rslora,
        )
        self.model = get_peft_model(self.model, lora_config)
        self._log_trainable_parameters()

    def _setup_unsloth(self):
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError("Unsloth backend selected but `unsloth` is not installed.") from exc

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            use_gradient_checkpointing="unsloth",
            use_rslora=self.config.use_rslora,
        )
        self._log_trainable_parameters()

    def _log_trainable_parameters(self):
        if not self.model:
            return
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "Trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )

    def train(self, train_dataset, eval_dataset=None) -> tuple[Any, TrainingSummary]:
        """Run QLoRA training and emit a summary compatible with agent_loop."""
        from finetune.zeroth_core import pre_train_zeroth_check
        import torch
        from trl import SFTConfig
        from finetune.safe_trainer import SafeQLoRATrainer
        from pathlib import Path

        # --- Zeroth Seam: Prevent unauthorized fine-tuning ---
        sample = [train_dataset[i] for i in range(min(5, len(train_dataset)))] if train_dataset else []
        from dataclasses import asdict

        job_id = f"train-{Path(self.config.output_dir).name}"
        pre_train_zeroth_check(asdict(self.config), sample, job_id=job_id)
        # -----------------------------------------------------

        total_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            max_seq_length=self.config.max_seq_length,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": train_dataset,
            "args": training_args,
            "dataset_text_field": "text",
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        safety_path = Path("datasets/zeroth_safety_golden.jsonl")
        safety_dataset = None
        if safety_path.exists():
            from finetune.data_formats import apply_alpaca_format

            s_records = load_jsonl_records(safety_path)
            texts = [apply_alpaca_format(r) for r in s_records]
            encoded = self.tokenizer(
                texts, padding="max_length", truncation=True, max_length=self.config.max_seq_length, return_tensors="pt"
            )

            s_dataset_list = []
            for i in range(len(texts)):
                s_dataset_list.append(
                    {
                        "input_ids": encoded["input_ids"][i],
                        "attention_mask": encoded["attention_mask"][i],
                        "labels": encoded["input_ids"][i].clone(),
                    }
                )
            safety_dataset = s_dataset_list

        trainer = SafeQLoRATrainer(**trainer_kwargs, safety_dataset=safety_dataset)

        train_t0 = time.time()
        train_result = trainer.train()
        training_seconds = time.time() - train_t0

        metrics: dict[str, float] = {}
        training_loss = getattr(train_result, "training_loss", None)
        if isinstance(training_loss, (int, float)):
            metrics["train_loss"] = float(training_loss)

        if eval_dataset is not None:
            for key, value in trainer.evaluate().items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            eval_loss = metrics.get("eval_loss")
            if isinstance(eval_loss, float) and eval_loss < 20:
                metrics["eval_perplexity"] = float(math.exp(eval_loss))

        primary_value = metrics.get(self.config.primary_metric)
        if primary_value is None:
            raise RuntimeError(f"Primary metric `{self.config.primary_metric}` was not produced.")

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        summary = TrainingSummary(
            primary_metric_name=self.config.primary_metric,
            primary_metric_value=float(primary_value),
            metric_goal=self.config.metric_goal,
            metrics=metrics,
            peak_vram_mb=peak_vram_mb,
            training_seconds=training_seconds,
            total_seconds=time.time() - total_t0,
        )
        return trainer, summary

    def save(self, path: str):
        """Save LoRA adapter weights."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved to %s", path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid QLoRA fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--backend", choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--dataset", help="Optional dataset path override (file or dir)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Build an eval split and emit eval metrics",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write the structured training summary JSON",
    )
    return parser.parse_args(argv)


def run_from_config(
    config_path: str | Path,
    backend: str | None = None,
    dataset_path: str | None = None,
    do_eval: bool = False,
) -> TrainingSummary:
    config = QLoRAConfig.from_yaml(Path(config_path))
    if backend:
        config.backend = backend
    if dataset_path:
        config.dataset_path = dataset_path

    if not config.dataset_path:
        raise ValueError("QLoRAConfig.dataset_path must be set for CLI execution.")

    records = load_jsonl_records(Path(config.dataset_path))
    eval_enabled = do_eval or config.primary_metric.startswith("eval_")
    split_ratio = config.eval_split_ratio if eval_enabled else 0.0
    train_dataset, eval_dataset = build_text_datasets(
        records,
        dataset_format=config.dataset_format,
        eval_split_ratio=split_ratio,
    )

    trainer = QLoRATrainer(config)
    trainer.setup()
    _, summary = trainer.train(train_dataset, eval_dataset=eval_dataset)
    trainer.save(config.output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    summary = run_from_config(
        config_path=args.config,
        backend=args.backend,
        dataset_path=args.dataset,
        do_eval=args.eval,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    for line in summary.to_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
