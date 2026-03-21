# TuneForge Private Pilot Runbook
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: PRIVATE_PILOT_RUNBOOK-EN.md

## Ziel

Reproduzierbare End-to-End-Validation-Evidence fuer TuneForge sammeln auf:

- Tier A: RTX 3090 / 24 GB
- Tier B: 48 GB+

## Checkliste fuer das Tester-Paket

- Environment-Checkliste
- Command Log
- Artefaktverzeichnis
- Environment Manifest
- Tester Attestation
- Failure Report, falls anwendbar

## Befehlsfolge

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

## Verbindlicher Einreichungsinhalt

- GPU-Modell und VRAM
- Treiber- und CUDA-Version
- Betriebssystem
- Git SHA
- exakter Config-Pfad
- Laufzeit und Peak VRAM
- Release-Bundle mit Validation Result

## Zaehlt nur dann fuer oeffentlichen Proof

Ein Run zaehlt nur, wenn:

- das Artefaktverzeichnis sauber validiert
- der Run-Status `passed` ist
- die Umgebung unabhaengig ist
- Tier und Modell zum Registry-Eintrag passen
- das Governance-Review die Evidence akzeptiert
