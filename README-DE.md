# TuneForge
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: README.md

[![Status](https://img.shields.io/badge/status-Technical%20Preview-orange)](docs/VALIDATION_MATRIX-DE.md)
[![Lizenz](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

TuneForge ist ein Open-Source-Framework fuer Self-Finetuning, Benchmarking und Modell-Publishing auf Basis des `karpathy/autoresearch`-Gedankens und wird fuer ein eigenes oeffentliches GitHub-Repo vorbereitet.

Aktueller oeffentlicher Status: **Technical Preview**.

TuneForge ist als auditfaehiges Engineering-Paket fuer lokale und kontrollierte Modellanpassung gedacht. Es ist Oesterreich-aware, DSGVO-aware und EU-AI-Act-aware. Es ist **keine Rechtsberatung** und garantiert **keine** regulatorische Konformitaet.

English overview: [README.md](README.md)  
Oeffentlicher Repo-Index: [REPO_INDEX-DE.md](REPO_INDEX-DE.md)

## Was dieses Repo liefert

- benchmark-first Research- und Fine-Tune-Workflows
- Release-Bundles fuer Hugging Face und Ollama-kompatible Auslieferung
- Validation Registry und Private-Pilot-Evidenzmodell
- zweisprachige Governance-, Trainings- und Compliance-Dokumentation
- oeffentliche Templates fuer Model Card, Provenance, Risk Review und Release Approval

## Schnellstart

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q tests
```

Docker-Preview-Oberflaeche:

```bash
docker compose -f docker-compose.finetune.yml up --build
```

Kanonisches Image-Ziel:

- `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- Images werden von GitHub Actions gebaut und publiziert, niemals ins Git committed

## Oeffentliche Repo-Regeln

- keine gebauten Docker-Images im Git
- keine Secrets oder echten `.env`-Dateien im Git
- keine generierten Modellartefakte im Git
- keine unklaren Compliance- oder Verifikationsclaims
- keine verifizierten Hardware-Labels ohne Registry-Evidenz

Beispielvariablen liegen in [.env.example](.env.example). Third-Party-Attribution liegt in [THIRD_PARTY.md](THIRD_PARTY.md) und [FORK.md](FORK.md).

## Kerndokumentation

- Repo-Map: [REPO_INDEX-DE.md](REPO_INDEX-DE.md)
- Architektur: [docs/ARCHITECTURE-DE.md](docs/ARCHITECTURE-DE.md)
- Validation Matrix: [docs/VALIDATION_MATRIX-DE.md](docs/VALIDATION_MATRIX-DE.md)
- Setup: [docs/SETUP-DE.md](docs/SETUP-DE.md)
- Training SOP: [docs/TRAINING_SOP-DE.md](docs/TRAINING_SOP-DE.md)
- Modell-Dokumentations-SOP: [docs/MODEL_DOCUMENTATION_SOP-DE.md](docs/MODEL_DOCUMENTATION_SOP-DE.md)
- Logging- und Audit-Protokoll: [docs/LOGGING_AUDIT_PROTOCOL-DE.md](docs/LOGGING_AUDIT_PROTOCOL-DE.md)
- Compliance Pack: [docs/COMPLIANCE_PACK-DE.md](docs/COMPLIANCE_PACK-DE.md)
- Upgrade-Plan: [docs/UPGRADE_PLAN-DE.md](docs/UPGRADE_PLAN-DE.md)
- Patch-Plan: [docs/PATCH_PLAN-DE.md](docs/PATCH_PLAN-DE.md)
- Private-Pilot-Runbook: [validation/PRIVATE_PILOT_RUNBOOK-DE.md](validation/PRIVATE_PILOT_RUNBOOK-DE.md)

## Governance-Position

TuneForge-Releases muessen durch Evidenz begrenzt bleiben:

- Preview, bis die Validation Registry die deklarierten Hardware-Tiers belegt
- Benchmark-Claims nur auf vergleichbaren Hardware-Budgets
- Release-Bundles muessen Manifeste, Provenance und Limitierungen enthalten
- Legal-/Compliance-Doks unterstuetzen Engineering-Review und Governance-Vorbereitung
- oeffentliche Aussagen muessen zu Oesterreich-, DSGVO- und EU-AI-Act-Framing passen, ohne Rechtszertifizierung zu behaupten

## Referenziertes Legal-Data-Subsystem

TuneForge dokumentiert `legal-scraper` als referenziertes Subsystem fuer Legal-Source-Ingestion und Provenance-Unterstuetzung. Es wird in dieser Phase nicht in den Produktcode vendored.

Siehe: [docs/LEGAL_SOURCE_REFERENCES-DE.md](docs/LEGAL_SOURCE_REFERENCES-DE.md)

## CI/CD

TuneForge nutzt GitHub Actions:

- Quality- und Public-Repo-Readiness in `.github/workflows/tuneforge-ci.yml`
- Preview-Docker-Releases in `.github/workflows/tuneforge-release.yml`
- Model-Bundle- und Hugging-Face-Publishing in `.github/workflows/tuneforge-model-publish.yml`

Release-Automation haengt SBOMs, Checksummen, Validation-Registry-Snapshots und Release-Metadaten an. Secrets liegen nur in GitHub Secrets oder einem externen Vault, nie im Repo.

## Attribution

TuneForge baut auf:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- `transformers`
- `trl`
- `peft`
- `unsloth`
- Hugging Face Hub
- `llama.cpp`
- Ollama

Details:

- [THIRD_PARTY.md](THIRD_PARTY.md)
- [FORK.md](FORK.md)
- [docs/CREDITS.md](docs/CREDITS.md)
- [docs/REFERENCES.md](docs/REFERENCES.md)

## Support-Flaeche

- Contribution Guide: [CONTRIBUTING-DE.md](CONTRIBUTING-DE.md)
- Support Guide: [SUPPORT-DE.md](SUPPORT-DE.md)
- Security Policy: [SECURITY-DE.md](SECURITY-DE.md)
- Compliance Statement: [COMPLIANCE_STATEMENT-DE.md](COMPLIANCE_STATEMENT-DE.md)
- Changelog: [CHANGELOG-DE.md](CHANGELOG-DE.md)

Der TuneForge-Quellcode in diesem Produktordner bleibt unter der [MIT License](LICENSE).
