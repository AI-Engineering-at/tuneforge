# Setup-Anleitung - TuneForge

## Geltungsbereich

Diese Anleitung beschreibt den oeffentlichen TuneForge-Setup fuer:

- kurze autonome Research-Runs
- hybrides QLoRA-Fine-Tuning
- Docker-First-Betrieb ueber TuneForge Studio

## Voraussetzungen

- Python 3.10+
- NVIDIA-GPU mit aktuellem Treiber
- Docker Desktop oder Docker Engine mit NVIDIA Container Toolkit
- Git

## Lokales Python-Setup

```bash
git clone https://github.com/AI-Engineerings-at/Playbook01.git
cd Playbook01/products/tuneforge
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[llm,finetune,dev]
python -m pytest -q
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

## Erster Fine-Tune-Lauf

```bash
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval
python -m finetune.trainer --config finetune/configs/sps-plc-qwen3.yaml --backend unsloth --eval
```

## Benchmark-Disziplin

- zuerst immer die Control-Config laufen lassen
- Kandidaten nur auf gleichem Hardware-Budget vergleichen
- nur dokumentierte Siege veroeffentlichen

## Governance-Checkliste

Vor einer oeffentlichen Veroeffentlichung pruefen:

- [RELEASE_POLICY.md](RELEASE_POLICY.md)
- [MODEL_RELEASE_POLICY.md](MODEL_RELEASE_POLICY.md)
- [COMPLIANCE_PACK.md](COMPLIANCE_PACK.md)
- [../THIRD_PARTY.md](../THIRD_PARTY.md)

## CI-Oberflaeche

Der Produkt-Workflow liegt unter `.github/workflows/tuneforge-ci.yml`.
