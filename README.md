# TuneForge
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: README-DE.md

[![Based on karpathy/autoresearch](https://img.shields.io/badge/based%20on-karpathy%2Fautoresearch-blue)](https://github.com/karpathy/autoresearch)
[![Docker](https://img.shields.io/badge/docker-TuneForge%20Studio-2496ED)](https://docs.docker.com/get-docker/)
[![Status](https://img.shields.io/badge/status-Technical%20Preview-orange)](docs/VALIDATION_MATRIX-EN.md)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

TuneForge is an open-source self-finetune and benchmark framework built on the spirit of `karpathy/autoresearch` and prepared for a dedicated public GitHub repo.

Current public status: **Technical Preview**.

TuneForge is an audit-ready engineering package for local and governed model adaptation on hardware teams actually own. It is Austria-aware, DSGVO-aware, and EU AI Act-aware. It is **not legal advice** and it does **not** guarantee regulatory compliance.

German overview: [README-DE.md](README-DE.md)  
Public repo index: [REPO_INDEX.md](REPO_INDEX.md)

## What This Repo Exposes

- benchmark-first research and fine-tune workflows
- release bundles for Hugging Face and Ollama-compatible publication
- validation registry and private-pilot evidence model
- bilingual governance, training, and compliance documentation
- public templates for model cards, provenance, risk review, and release approval

## Quickstart

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q tests
```

Docker preview surface:

```bash
docker compose -f docker-compose.finetune.yml up --build
```

Canonical image target:

- `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- images are built and published by GitHub Actions, never committed to git

## Public Repo Rules

- no built Docker images in git
- no secrets or real `.env` files in git
- no generated model artifacts in git
- no ambiguous compliance or verification claims
- no verified hardware label without registry proof

Sample environment variables live in [.env.example](.env.example). Third-party attribution lives in [THIRD_PARTY.md](THIRD_PARTY.md) and [FORK.md](FORK.md).

## Core Documentation

- repo map: [REPO_INDEX.md](REPO_INDEX.md)
- architecture: [docs/ARCHITECTURE-EN.md](docs/ARCHITECTURE-EN.md)
- validation matrix: [docs/VALIDATION_MATRIX-EN.md](docs/VALIDATION_MATRIX-EN.md)
- setup: [docs/SETUP-EN.md](docs/SETUP-EN.md)
- training SOP: [docs/TRAINING_SOP-EN.md](docs/TRAINING_SOP-EN.md)
- model documentation SOP: [docs/MODEL_DOCUMENTATION_SOP-EN.md](docs/MODEL_DOCUMENTATION_SOP-EN.md)
- logging and audit protocol: [docs/LOGGING_AUDIT_PROTOCOL-EN.md](docs/LOGGING_AUDIT_PROTOCOL-EN.md)
- compliance pack: [docs/COMPLIANCE_PACK-EN.md](docs/COMPLIANCE_PACK-EN.md)
- upgrade plan: [docs/UPGRADE_PLAN-EN.md](docs/UPGRADE_PLAN-EN.md)
- patch plan: [docs/PATCH_PLAN-EN.md](docs/PATCH_PLAN-EN.md)
- private pilot runbook: [validation/PRIVATE_PILOT_RUNBOOK-EN.md](validation/PRIVATE_PILOT_RUNBOOK-EN.md)

## Governance Position

TuneForge public releases must remain bounded by evidence:

- preview until the validation registry proves the declared hardware tiers
- benchmark claims only on matched hardware budgets
- release bundles must include manifests, provenance, and limitations
- legal/compliance docs support engineering review and governance preparation
- public statements must remain compatible with Austria, DSGVO, and EU AI Act framing without pretending to be legal certification

## Referenced Legal Data Subsystem

TuneForge documents `legal-scraper` as a referenced subsystem for legal source ingestion and provenance support. It is not vendored into the product code in this phase.

See: [docs/LEGAL_SOURCE_REFERENCES-EN.md](docs/LEGAL_SOURCE_REFERENCES-EN.md)

## CI/CD

TuneForge CI/CD is GitHub Actions based:

- quality and public repo readiness in `.github/workflows/tuneforge-ci.yml`
- preview Docker releases in `.github/workflows/tuneforge-release.yml`
- model bundle and Hugging Face publishing in `.github/workflows/tuneforge-model-publish.yml`

Release automation attaches SBOMs, checksums, validation registry snapshots, and release metadata. Secrets live only in GitHub Secrets or an external vault, never in the repo.

## Attribution

TuneForge is built with and on top of:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- `transformers`
- `trl`
- `peft`
- `unsloth`
- Hugging Face Hub
- `llama.cpp`
- Ollama

Detailed attribution:

- [THIRD_PARTY.md](THIRD_PARTY.md)
- [FORK.md](FORK.md)
- [docs/CREDITS.md](docs/CREDITS.md)
- [docs/REFERENCES.md](docs/REFERENCES.md)

## Support Surface

- contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- support guide: [SUPPORT.md](SUPPORT.md)
- security policy: [SECURITY.md](SECURITY.md)
- compliance statement: [COMPLIANCE_STATEMENT.md](COMPLIANCE_STATEMENT.md)
- changelog: [CHANGELOG.md](CHANGELOG.md)

TuneForge source code in this product directory remains under the [MIT License](LICENSE).
