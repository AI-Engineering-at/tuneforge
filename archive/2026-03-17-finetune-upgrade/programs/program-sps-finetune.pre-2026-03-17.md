# AutoResearch Program: SPS/PLC Fine-Tuning Optimization

## Goal
Optimize QLoRA fine-tuning hyperparameters for an IEC 61131-3 Structured Text
code generation model. Maximize compile rate and semantic accuracy on the
validation set.

## What You Can Modify
- finetune/configs/sps-plc.yaml: ALL fields
  - lora_r (4, 8, 16, 32, 64)
  - lora_alpha (2x lora_r is common, try others)
  - learning_rate (1e-5 to 5e-4)
  - warmup_ratio (0.0 to 0.1)
  - gradient_accumulation_steps (1, 2, 4, 8, 16)
  - max_seq_length (1024, 2048, 4096)
  - weight_decay (0.0 to 0.1)
  - target_modules (which layers to tune)

## What You CANNOT Modify
- Base model (Qwen/Qwen2.5-Coder-7B-Instruct)
- Training data (datasets/generated/sps/)
- Evaluation script (finetune/evaluator.py)
- Quantization (4-bit NF4)

## Metrics
- PRIMARY: compile_rate (% of generated code that compiles)
- SECONDARY: semantic_accuracy (% of correct behavior)
- CONSTRAINT: peak_vram_mb < 22000 (must fit RTX 3090)

## Protocol
1. Change ONE parameter in sps-plc.yaml
2. Run: python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
3. Record: compile_rate, semantic_accuracy, peak_vram, training_time
4. If compile_rate improved: KEEP
5. If compile_rate same/worse: DISCARD (revert yaml)
6. NEVER STOP until interrupted
