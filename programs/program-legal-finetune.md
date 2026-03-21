# AutoResearch Program: Legal/DSGVO Fine-Tuning Optimization

## Goal
Optimize the legal QLoRA pipeline for German DSGVO-style text while keeping the
current control config intact until a newer benchmark candidate proves better on
the same 3090 budget.

## Configs
- Control: `finetune/configs/legal-dsgvo.yaml`
- Qwen3 candidate: `finetune/configs/legal-dsgvo-qwen3.yaml`
- Optional backend override: `--backend peft_trl` or `--backend unsloth`

## What You Can Modify
- Any training hyperparameter in the selected config file
- `backend`
- `lora_r`, `lora_alpha`, `lora_dropout`, `use_rslora`
- `learning_rate`, `warmup_ratio`, `gradient_accumulation_steps`
- `max_seq_length`, `weight_decay`
- `target_modules`

## What You Must Preserve
- Benchmark-first rollout: do not replace the control config without measured evidence
- Quantization stays 4-bit NF4
- `target_modules` keep the full 7-layer set unless you are explicitly testing a regression
- `lora_dropout` should stay at `0.0` unless benchmark data justifies changing it

## Metrics
- PRIMARY: `eval_loss` from `python -m finetune.trainer`
- SECONDARY: `train_loss`, `eval_perplexity`, `peak_vram_mb`, `training_seconds`
- CONSTRAINT: `peak_vram_mb < 22000`

## Protocol
1. Establish the control result with `legal-dsgvo.yaml`.
2. Run the same experiment budget on `legal-dsgvo-qwen3.yaml`.
3. Optionally rerun the winning config with `--backend unsloth`.
4. Keep only changes that improve `eval_loss` while fitting on a 3090.
5. Track `Ministral 3 8B` as a later text-only compatibility candidate.
6. Do not add `Mistral Small 4` here. It is documentation-only for this phase.

## Commands
```bash
python -m finetune.trainer --config finetune/configs/legal-dsgvo.yaml --eval
python -m finetune.trainer --config finetune/configs/legal-dsgvo-qwen3.yaml --eval
python -m finetune.trainer --config finetune/configs/legal-dsgvo-qwen3.yaml --backend unsloth --eval
```
