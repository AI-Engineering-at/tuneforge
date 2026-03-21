# Setup Guide - TuneForge

## Scope

This guide covers the public TuneForge setup for:

- autonomous short-budget research runs
- hybrid QLoRA fine-tuning
- Docker-first execution through TuneForge Studio

## Prerequisites

- Python 3.10+
- NVIDIA GPU and recent driver
- Docker Desktop or Docker Engine with NVIDIA Container Toolkit for container runs
- Git

## Local Python Setup

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q
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

## First Fine-Tune Run

```bash
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
```

## Benchmark Discipline

- run the control first
- compare candidates on the same hardware budget
- publish only documented wins

## Governance Checklist

Before public release, check:

- [RELEASE_POLICY.md](RELEASE_POLICY.md)
- [MODEL_RELEASE_POLICY.md](MODEL_RELEASE_POLICY.md)
- [COMPLIANCE_PACK.md](COMPLIANCE_PACK.md)
- [../THIRD_PARTY.md](../THIRD_PARTY.md)

## CI Surface

The product CI workflow is `.github/workflows/tuneforge-ci.yml`.
