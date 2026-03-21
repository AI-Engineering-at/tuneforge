# AutoResearch Program: SPS/PLC Fine-Tuning Optimization

## Goal
Optimize the SPS/PLC QLoRA pipeline for IEC 61131-3 Structured Text while keeping
the current control config intact until a benchmark candidate clearly wins on a
single RTX 3090.

## Configs
- Control: `finetune/configs/sps-plc.yaml`
- Qwen3 candidate: `finetune/configs/sps-plc-qwen3.yaml`
- Optional backend override: `--backend peft_trl` or `--backend unsloth`

## What You Can Modify
- Any training hyperparameter in the selected config file
- `backend`
- `lora_r`, `lora_alpha`, `lora_dropout`, `use_rslora`
- `learning_rate`, `warmup_ratio`, `gradient_accumulation_steps`
- `max_seq_length`, `weight_decay`
- `target_modules`

## What You Must Preserve
- Benchmark-first rollout: never replace the control config without evidence
- Quantization stays 4-bit NF4
- `target_modules` keep the full 7-layer set unless you are explicitly testing a regression
- `lora_dropout` should stay at `0.0` unless you are measuring a reason to increase it

## Metrics
- PRIMARY: `eval_loss` from `python -m finetune.trainer`
- SECONDARY: `train_loss`, `eval_perplexity`, `peak_vram_mb`, `training_seconds`
- CONSTRAINT: `peak_vram_mb < 22000`

## Protocol
1. Establish the control result with `sps-plc.yaml`.
2. Run the same experiment budget on `sps-plc-qwen3.yaml`.
3. Optionally rerun the winning config with `--backend unsloth`.
4. Keep only changes that improve `eval_loss` while fitting on a 3090.
5. Do not add `Mistral Small 4` here. It is out of scope for this local path.

## Commands
```bash
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
```
