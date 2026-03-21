# TuneForge Troubleshooting
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: TROUBLESHOOTING-EN.md

## GPU wird nicht erkannt

- `nvidia-smi` pruefen
- NVIDIA Container Toolkit fuer Docker-Runs pruefen
- bestaetigen, dass der Container mit GPU-Zugriff gestartet wurde
- pruefen, ob Treiber- und CUDA-Version im Environment Manifest erfasst sind

## Unsloth-Backend fehlt

- das `finetune`-Extra installieren
- bestaetigen, dass die Umgebung wirklich `unsloth` enthaelt
- im Zweifel `peft_trl` fuer die Control-Benchmarks verwenden

## Primaere Metrik fehlt

- `primary_metric` und `metric_goal` in der YAML-Config pruefen
- bestaetigen, dass `--eval` aktiv ist, wenn die Metrik aus der Evaluation stammt
- `results/protocol.jsonl` auf die emittierte Metrikfolge pruefen

## Release-Bundle-Validierung schlaegt fehl

- `training-manifest.json`, `benchmark-summary.json`, `environment-manifest.json` und `tester-attestation.json` pruefen
- `python scripts/validate_release_artifacts.py <bundle-dir>` ausfuehren
- `validation-result.json` inspizieren
- pruefen, ob Public Status und Hardware Tier im Bundle konsistent sind

## Public-Repo-Hygiene schlaegt fehl

- generierte Outputs aus dem tracked state entfernen
- nur `.env.example`, niemals `.env`, versionieren
- Docker-Images, GGUF, Safetensors, Logs und Datenbanken nicht in Git halten
- `python scripts/check_repo_hygiene.py` ausfuehren

## Docs-Parity schlaegt fehl

- EN und DE gemeinsam aktualisieren
- `Language`, `Audience`, `Last Sync` und `Pair` synchron halten
- `python scripts/check_docs_parity.py` ausfuehren

## Public-Claim-Check schlaegt fehl

- nicht verifizierte Hardware-Labels entfernen
- `Technical Preview` in den Root-Readmes sichtbar halten, solange die Registry Preview ist
- Sprache wie compliance-certified oder ready-for-production vermeiden
- `python scripts/check_public_claims.py` ausfuehren
