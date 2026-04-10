# Trainer API

The `QLoRATrainer` class is the main interface for fine-tuning.

## QLoRAConfig

Configuration class for QLoRA training.

```python
from finetune.trainer import QLoRAConfig

config = QLoRAConfig(
    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    output_dir="output/my-model",
    backend="peft_trl",  # or "unsloth"
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    max_steps=1000,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | str | `"Qwen/Qwen2.5-Coder-7B-Instruct"` | HuggingFace model ID |
| `output_dir` | str | `"output"` | Output directory |
| `backend` | str | `"peft_trl"` | Training backend |
| `dataset_path` | str | `""` | Path to dataset |
| `dataset_format` | str | `"alpaca"` | Dataset format |
| `lora_r` | int | 16 | LoRA rank |
| `lora_alpha` | int | 32 | LoRA alpha |
| `learning_rate` | float | 2e-4 | Learning rate |
| `max_steps` | int | 1000 | Maximum training steps |

## QLoRATrainer

Main trainer class.

```python
from finetune.trainer import QLoRATrainer, QLoRAConfig

config = QLoRAConfig(...)
trainer = QLoRATrainer(config)
```

### Methods

#### train()

Run training.

```python
trainer.train(dataset_path="data/train.jsonl")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_path` | str | Path to training dataset |

**Returns:** `TrainingSummary`

## TrainingSummary

Summary of training results.

```python
from finetune.trainer import TrainingSummary

summary = trainer.train(...)
print(summary.primary_metric_value)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `primary_metric_name` | str | Name of primary metric |
| `primary_metric_value` | float | Value of primary metric |
| `peak_vram_mb` | float | Peak VRAM usage |
| `training_seconds` | float | Training duration |

### Methods

#### to_dict()

Convert to dictionary.

```python
data = summary.to_dict()
```

#### to_lines()

Convert to formatted lines.

```python
lines = summary.to_lines()
```

## Data Loading

### load_jsonl_records()

Load records from JSONL file.

```python
from finetune.trainer import load_jsonl_records

records = load_jsonl_records("data/train.jsonl")
```

### build_text_datasets()

Build train/eval datasets.

```python
from finetune.trainer import build_text_datasets

train_ds, eval_ds = build_text_datasets(
    records=records,
    dataset_format="alpaca",
    eval_split_ratio=0.1
)
```

## Example

```python
from finetune.trainer import QLoRAConfig, QLoRATrainer

# Configure
config = QLoRAConfig(
    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    output_dir="output/my-model",
    backend="peft_trl",
    max_steps=100,
)

# Train
trainer = QLoRATrainer(config)
summary = trainer.train(dataset_path="data/train.jsonl")

# Print results
print(f"Eval loss: {summary.primary_metric_value}")
print(f"Peak VRAM: {summary.peak_vram_mb} MB")
```
