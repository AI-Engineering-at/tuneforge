# TuneForge Private Pilot Runbook
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: PRIVATE_PILOT_RUNBOOK-DE.md

## Goal

Collect reproducible end-to-end validation evidence for TuneForge on:

- Tier A: RTX 3090 / 24 GB
- Tier B: 48 GB+

## Tester Package Checklist

- environment checklist
- command log
- artifact directory
- environment manifest
- tester attestation
- failure report when applicable

## Command Sequence

```bash
python -m pytest -q tests
docker compose -f docker-compose.finetune.yml up --build
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
python -m finetune.model_publisher bundle \
  --adapter-path output/private-pilot-run \
  --model-name local/tuneforge-private-pilot \
  --base-model Qwen/Qwen3-8B \
  --language de \
  --domain "Industrial Automation / IEC 61131-3" \
  --training-data "Pilot benchmark data" \
  --dataset "Pilot benchmark data" \
  --primary-metric eval_loss \
  --metric-goal minimize \
  --primary-value 0.5 \
  --hardware "RTX 3090" \
  --hardware-tier tier_a_rtx_3090_24gb \
  --gpu-model "RTX 3090" \
  --gpu-vram-gb 24 \
  --driver-version "<driver>" \
  --cuda-version "<cuda>" \
  --tester-id "<tester-id>" \
  --tester-organization "<org>" \
  --submission-kind private_pilot \
  --result-status passed \
  --independent-environment \
  --base-model-license Apache-2.0 \
  --gguf-filename model.gguf \
  --limitation "Human review required."
python scripts/validate_release_artifacts.py output/private-pilot-run
```

## Required Submission Contents

- GPU model and VRAM
- driver and CUDA versions
- OS
- git SHA
- exact config path
- timing and peak VRAM
- release bundle with validation result

## Counting Toward Public Proof

A run counts only if:

- the artifact directory validates cleanly
- the run status is `passed`
- the environment is independent
- the tier and model match the registry entry
- governance review accepts the evidence
