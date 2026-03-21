# TuneForge

[![Based on karpathy/autoresearch](https://img.shields.io/badge/based%20on-karpathy%2Fautoresearch-blue)](https://github.com/karpathy/autoresearch)
[![Docker](https://img.shields.io/badge/docker-TuneForge%20Studio-2496ED)](https://docs.docker.com/get-docker/)
[![GPU](https://img.shields.io/badge/GPU-8GB%20to%20H100-green)](docs/GPU-TIERS.md)
[![Status](https://img.shields.io/badge/status-Technical%20Preview-orange)](docs/VALIDATION_MATRIX.md)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

TuneForge is an open-source self-finetune framework for reproducible model adaptation, benchmark-first experimentation, and model publishing on hardware teams actually own.

It keeps the `karpathy/autoresearch` spirit for short autonomous research loops and adds a governed QLoRA path for domain adapters, release bundles, Hugging Face publishing, and Ollama-compatible packaging.

`TuneForge Studio` is the Docker-first product surface of this repo.

Current public status: **Technical Preview**.

TuneForge does not claim verified hardware support or production readiness until the validation registry records enough independent successful runs on the documented hardware tiers.

## Four Paths

### 1. Quickstart

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q
```

### 2. Docker

TuneForge Studio is the official Docker entrypoint for self-finetuning and benchmark reproduction.

```bash
docker compose -f docker-compose.finetune.yml up --build
```

Release target:

- Canonical image target: `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- Phase 1 registry: GHCR only
- GPU support: single-GPU consumer and workstation tiers, including stronger enterprise hardware
- public Docker releases remain preview-tagged until the validation matrix is complete

### 3. Benchmarks

TuneForge ships with a benchmark-first policy:

- keep shipped control configs stable
- add newer models as candidates
- publish only documented wins on the same hardware budget
- never silently replace a control because a newer model exists

Current fine-tune candidates:

- SPS/PLC control: `Qwen/Qwen2.5-Coder-7B-Instruct`
- SPS/PLC candidate: `Qwen/Qwen3-8B`
- Legal/DSGVO control: `LeoLM/leo-mistral-hessianai-7b`
- Legal/DSGVO candidate: `Qwen/Qwen3-8B`
- `Mistral Small 4`: documentation-only, not a single-3090 target

### 4. Publish Models

TuneForge supports a release bundle per published adapter or GGUF:

- `README.md` / Model Card
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `environment-manifest.json`
- `tester-attestation.json`
- `validation-result.json`
- `Modelfile` for Ollama-compatible distribution

Publishing targets:

- GitHub Releases for source, Docker images, SBOMs, checksums, and benchmark artifacts
- Hugging Face for adapters, model cards, and public manifests
- Ollama-compatible packaging via GGUF plus `Modelfile`

## What TuneForge Does

TuneForge has two coordinated execution paths:

1. `Autonomous research loop`
   Uses an LLM agent to edit training code, run a fixed-budget experiment, parse results, and keep or discard changes.

2. `Hybrid fine-tune runtime`
   Uses `transformers + peft + trl` as the stable backend and optional `unsloth + trl` as the benchmark backend for domain adapters.

Both paths emit machine-readable metrics and follow the same benchmark discipline.

Hardware promotion, public claims, and model publication are gated by the validation registry under `validation/`.

## Architecture

```text
+--------------------- TuneForge OSS Surface ----------------------+
|                                                                  |
|  Source + Docs + Docker + Benchmarks + Release Manifests         |
|                                                                  |
|  Quickstart      Docker        Benchmarks        Publish Models  |
|                                                                  |
+-------------------------------+----------------------------------+
                                |
                                v
+------------------------ TuneForge Runtime -----------------------+
|                                                                  |
|  agent_loop.py      providers.py      finetune/trainer.py        |
|                                                                  |
|  - autonomous loop  - multi-provider  - PEFT/TRL control         |
|  - metric parser    - Claude/OpenAI   - Unsloth benchmark        |
|  - keep/discard     - Ollama/Gemini   - release artifacts        |
|                                                                  |
+-------------------------------+----------------------------------+
                                |
                                v
+---------------------- Release and Knowledge ---------------------+
|                                                                  |
|  GitHub Releases   Hugging Face   Ollama   open-notebook         |
|                                                                  |
|  Public: code, reports, adapters, GGUFs, manifests               |
|  Private: research notes, internal reviews, content strategy     |
|                                                                  |
+------------------------------------------------------------------+
```

## Hardware Tiers

| Tier | Hardware | Primary Use |
|------|----------|-------------|
| 8 GB | RTX 2060, 3060, 4060, laptop GPUs | short autoresearch runs, small local experiments |
| 24 GB | RTX 3090, RTX 4090 | primary single-GPU fine-tune and benchmark target, with RTX 3090 as the public reference tier |
| 48 GB+ | RTX 6000 Ada, A6000, A100, H100 | stronger enterprise benchmarking and larger batch budgets |

Full guidance: [docs/GPU-TIERS.md](docs/GPU-TIERS.md)

## Compliance and Governance

TuneForge is positioned as a DSGVO-aware and EU-AI-Act-ready engineering framework. That means:

- documented model provenance
- explicit intended use and out-of-scope use
- benchmark evidence for release claims
- validation registry evidence for hardware-verification labels
- license manifests and third-party attribution
- release gates before public publication

It does not provide legal advice. The public governance pack lives here:

- [docs/COMPLIANCE_PACK.md](docs/COMPLIANCE_PACK.md)
- [docs/RELEASE_POLICY.md](docs/RELEASE_POLICY.md)
- [docs/MODEL_RELEASE_POLICY.md](docs/MODEL_RELEASE_POLICY.md)
- [SECURITY.md](SECURITY.md)
- [THIRD_PARTY.md](THIRD_PARTY.md)

## Built With / Based On

TuneForge is built with and on top of:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the original autonomous training loop
- `transformers`, `peft`, and `trl` for the stable fine-tune backend
- `unsloth` for optional benchmark backend experiments
- Hugging Face Hub for adapter distribution
- `llama.cpp` / GGUF-compatible tooling for Ollama packaging
- Ollama for local inference distribution

Detailed attribution:

- [FORK.md](FORK.md)
- [docs/CREDITS.md](docs/CREDITS.md)
- [docs/REFERENCES.md](docs/REFERENCES.md)
- [THIRD_PARTY.md](THIRD_PARTY.md)

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Setup Guide (EN)](docs/SETUP-EN.md)
- [Setup Guide (DE)](docs/SETUP-DE.md)
- [GPU Tiers](docs/GPU-TIERS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Validation Matrix](docs/VALIDATION_MATRIX.md)
- [Use Cases](docs/USE_CASES.md)
- [Business ROI](docs/BUSINESS_ROI.md)
- [Release Policy](docs/RELEASE_POLICY.md)
- [Model Release Policy](docs/MODEL_RELEASE_POLICY.md)
- [Compliance Pack](docs/COMPLIANCE_PACK.md)
- [open-notebook Pipeline](docs/OPEN_NOTEBOOK_PIPELINE.md)
- [Private Pilot Runbook](validation/PRIVATE_PILOT_RUNBOOK.md)
- [Changelog](docs/CHANGELOG.md)
- [Fork Attribution](FORK.md)
- [Credits](docs/CREDITS.md)

## Release Discipline

Every public TuneForge release should be reproducible and auditable:

- semver tag for code and Docker image
- benchmark evidence attached to the release
- preview-only public status until the validation registry proves otherwise
- release bundle generated from the exact adapter or GGUF
- signed Docker image in GHCR
- archive snapshot before major branding or architecture shifts

## License

TuneForge source code in this product directory is released under the [MIT License](LICENSE), with third-party components remaining under their respective licenses. Upstream `karpathy/autoresearch` remains attributed and preserved under `upstream/`.
