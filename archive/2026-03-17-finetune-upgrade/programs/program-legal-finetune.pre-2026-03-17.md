# AutoResearch Program: Legal/DSGVO Fine-Tuning Optimization

## Goal
Optimize QLoRA fine-tuning hyperparameters for a German legal text model
specializing in DSGVO compliance and EU AI Act interpretation. Maximize
answer accuracy and legal citation quality.

## What You Can Modify
- finetune/configs/legal-dsgvo.yaml: ALL fields
  - lora_r (4, 8, 16, 32, 64)
  - lora_alpha (2x lora_r is common, try others)
  - learning_rate (1e-5 to 5e-4)
  - warmup_ratio (0.0 to 0.1)
  - gradient_accumulation_steps (1, 2, 4, 8, 16)
  - max_seq_length (1024, 2048, 4096)
  - weight_decay (0.0 to 0.1)
  - target_modules (which layers to tune)

## What You CANNOT Modify
- Base model (LeoLM/leo-mistral-hessianai-7b)
- Training data (datasets/generated/legal/)
- Evaluation script (finetune/evaluator.py)
- Quantization (4-bit NF4)

## Metrics
- PRIMARY: answer_accuracy (% of correct legal answers on test set)
- SECONDARY: citation_accuracy (% of correct article references)
- CONSTRAINT: peak_vram_mb < 22000 (must fit RTX 3090)

## Protocol
1. Change ONE parameter in legal-dsgvo.yaml
2. Run: python -m finetune.trainer --config finetune/configs/legal-dsgvo.yaml --eval
3. Record: answer_accuracy, citation_accuracy, peak_vram, training_time
4. If answer_accuracy improved: KEEP
5. If answer_accuracy same/worse: DISCARD (revert yaml)
6. NEVER STOP until interrupted
