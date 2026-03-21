# TuneForge Setup Guide
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: SETUP-DE.md

Current public status: **Technical Preview**.

## Purpose

This guide covers the public TuneForge setup for:

- autonomous short-budget research runs
- hybrid QLoRA fine-tuning
- Docker-first execution through TuneForge Studio
- release-bundle generation for preview validation and publication

## Prerequisites

- Python 3.10 or newer
- NVIDIA GPU with recent drivers
- Docker Desktop or Docker Engine with NVIDIA Container Toolkit for container runs
- Git
- sample environment values from [../.env.example](../.env.example)

## Local Python Setup

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q tests
```

## Docker Setup

Main product surface:

```bash
docker compose up --build
```

Fine-tune surface:

```bash
docker compose -f docker-compose.finetune.yml up --build
```

Docker images stay outside git. Public images are expected to be built and published by GitHub Actions to GHCR.

## First Local Fine-Tune Run

```bash
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
```

## First Release-Bundle Run

```bash
python -m finetune.model_publisher bundle ^
  --adapter-path output/local-preview ^
  --model-name local/tuneforge-preview ^
  --base-model Qwen/Qwen3-8B ^
  --language de ^
  --domain "Industrial Automation / IEC 61131-3" ^
  --training-data "Synthetic IEC data" ^
  --dataset "Synthetic IEC data" ^
  --primary-metric eval_loss ^
  --metric-goal minimize ^
  --primary-value 0.5 ^
  --hardware "RTX 3090" ^
  --hardware-tier tier_a_rtx_3090_24gb ^
  --gpu-model "RTX 3090" ^
  --gpu-vram-gb 24 ^
  --tester-id local-preview ^
  --submission-kind local_smoke ^
  --base-model-license Apache-2.0
python scripts/validate_release_artifacts.py output/local-preview
```

## Benchmark Discipline

- run the shipped control first
- compare candidates only on the same hardware budget
- keep 3090 as the primary public reference tier
- do not claim verified hardware support before the registry proves it

## Governance Checks Before Public Release

- [ARCHITECTURE-EN.md](ARCHITECTURE-EN.md)
- [VALIDATION_MATRIX-EN.md](VALIDATION_MATRIX-EN.md)
- [RELEASE_POLICY-EN.md](RELEASE_POLICY-EN.md)
- [MODEL_RELEASE_POLICY-EN.md](MODEL_RELEASE_POLICY-EN.md)
- [COMPLIANCE_PACK-EN.md](COMPLIANCE_PACK-EN.md)
- [TRAINING_SOP-EN.md](TRAINING_SOP-EN.md)
- [MODEL_DOCUMENTATION_SOP-EN.md](MODEL_DOCUMENTATION_SOP-EN.md)
- [LOGGING_AUDIT_PROTOCOL-EN.md](LOGGING_AUDIT_PROTOCOL-EN.md)

## CI Surface

The product CI workflow is [../../../.github/workflows/tuneforge-ci.yml](../../../.github/workflows/tuneforge-ci.yml).
