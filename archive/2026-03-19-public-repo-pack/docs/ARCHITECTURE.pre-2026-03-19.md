# Architecture

## Overview

TuneForge has two coordinated product paths:

1. the original autonomous research loop for short, fixed-budget experiments
2. the hybrid fine-tune path for releaseable domain adapters

The repo does not treat fine-tuning as a side script. Both paths feed the same metric discipline and public release surface.

Current public operating mode: `Technical Preview` until the validation registry proves the declared hardware tiers.

## Runtime Layers

### Upstream research loop

- Source of truth: `upstream/`
- Downstream execution surface: `train.py`, `patches/`, `configs/`, `programs/`, `entrypoint.sh`
- Primary metric contract: legacy `val_bpb`

### Provider abstraction

- Source of truth: `providers.py`
- Supported families: Anthropic, OpenAI-compatible endpoints, Gemini, Ollama
- Used by the autonomous loop and synthetic data generation

### Autonomous orchestration

- Source of truth: `agent_loop.py`, `agent_config.py`
- Reads a program, proposes one change, runs one experiment, and compares a configurable primary metric

### Hybrid fine-tune runtime

- Source of truth: `finetune/trainer.py`
- Stable backend: `peft_trl`
- Optional benchmark backend: `unsloth`
- Public config contract:
  - `backend`
  - `dataset_path`
  - `dataset_format`
  - `primary_metric`
  - `metric_goal`
  - `use_rslora`

### Release and publishing layer

- Source of truth: `finetune/model_publisher.py`
- Generates model cards, training manifests, benchmark summaries, environment manifests, tester attestations, and Ollama `Modelfile` assets
- Publishes adapters to Hugging Face and prepares Ollama-compatible packaging

### Validation and proof layer

- Source of truth: `validation/registry.json`, `validation/PRIVATE_PILOT_RUNBOOK.md`
- Enforces hardware proof before public hardware-verification labels are allowed
- Treats RTX 3090 as the primary public reference tier and 48 GB+ as the secondary verified tier

### Governance layer

- Source of truth: `README.md`, `THIRD_PARTY.md`, `SECURITY.md`, `docs/RELEASE_POLICY.md`, `docs/MODEL_RELEASE_POLICY.md`, `docs/COMPLIANCE_PACK.md`, `docs/VALIDATION_MATRIX.md`
- Defines what can be published, under what evidence, and with which attribution

## Rollout Policy

TuneForge follows benchmark-first rollout:

- controls stay stable
- candidates are explicit
- promotions require a documented win on the same hardware budget
- public hardware-verification labels require at least two independent successful runs per tier

Current control/candidate split:

- SPS/PLC control: `Qwen/Qwen2.5-Coder-7B-Instruct`
- SPS/PLC candidate: `Qwen/Qwen3-8B`
- Legal/DSGVO control: `LeoLM/leo-mistral-hessianai-7b`
- Legal/DSGVO candidate: `Qwen/Qwen3-8B`

## Knowledge Flow

- internal source of truth: open-notebook
- public exports: docs, benchmark reports, model cards, blog posts, release notes
- validation evidence: private pilot submissions plus the public validation registry
- archive snapshots: `archive/`
