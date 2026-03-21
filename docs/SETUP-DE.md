# TuneForge Setup-Anleitung
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: SETUP-EN.md

Der aktuelle oeffentliche Status ist **Technical Preview**.

## Zweck

Diese Anleitung beschreibt das oeffentliche TuneForge-Setup fuer:

- kurze autonome Research-Runs
- hybrides QLoRA-Fine-Tuning
- Docker-First-Betrieb ueber TuneForge Studio
- Release-Bundle-Erzeugung fuer Preview-Validierung und Publikation

## Voraussetzungen

- Python 3.10 oder neuer
- NVIDIA-GPU mit aktuellem Treiber
- Docker Desktop oder Docker Engine mit NVIDIA Container Toolkit
- Git
- Beispielwerte aus [../.env.example](../.env.example)

## Lokales Python-Setup

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q tests
```

## Docker-Setup

Hauptoberflaeche:

```bash
docker compose up --build
```

Fine-Tune-Oberflaeche:

```bash
docker compose -f docker-compose.finetune.yml up --build
```

Docker-Images bleiben ausserhalb von Git. Oeffentliche Images sollen durch GitHub Actions nach GHCR gebaut und publiziert werden.

## Erster lokaler Fine-Tune-Lauf

```bash
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
```

## Erster Release-Bundle-Lauf

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

## Benchmark-Disziplin

- zuerst immer die ausgelieferte Control-Konfiguration laufen lassen
- Kandidaten nur auf gleichem Hardware-Budget vergleichen
- die 3090 als primaere oeffentliche Referenzplattform behandeln
- keinen verifizierten Hardware-Support claimen, bevor die Registry ihn belegt

## Governance-Checks vor oeffentlicher Publikation

- [ARCHITECTURE-DE.md](ARCHITECTURE-DE.md)
- [VALIDATION_MATRIX-DE.md](VALIDATION_MATRIX-DE.md)
- [RELEASE_POLICY-DE.md](RELEASE_POLICY-DE.md)
- [MODEL_RELEASE_POLICY-DE.md](MODEL_RELEASE_POLICY-DE.md)
- [COMPLIANCE_PACK-DE.md](COMPLIANCE_PACK-DE.md)
- [TRAINING_SOP-DE.md](TRAINING_SOP-DE.md)
- [MODEL_DOCUMENTATION_SOP-DE.md](MODEL_DOCUMENTATION_SOP-DE.md)
- [LOGGING_AUDIT_PROTOCOL-DE.md](LOGGING_AUDIT_PROTOCOL-DE.md)

## CI-Oberflaeche

Der Produkt-CI-Workflow liegt unter [../../../.github/workflows/tuneforge-ci.yml](../../../.github/workflows/tuneforge-ci.yml).
