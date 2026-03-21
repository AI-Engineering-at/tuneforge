# TuneForge Architektur
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: ARCHITECTURE-EN.md

Der aktuelle oeffentliche Status ist **Technical Preview**.

## Architektonische Position

TuneForge ist ein geschichtetes Upgrade auf `karpathy/autoresearch`, kein Rewrite.

- `autoresearch` bleibt die Grundlage fuer den Research-Loop
- TuneForge ergaenzt lokales Fine-Tuning, governte Publikation, Validation-Evidence und Public-Repo-Disziplin
- das primaere oeffentliche Ziel ist Single-GPU RTX 3090 zuerst, mit 48 GB+ als zweitem verifizierten Tier

## Laufzeitschichten

### Research-Loop

- Source of Truth: `agent_loop.py`, `agent_config.py`, `providers.py`
- Zweck: Experimente vorschlagen, ausfuehren, vergleichen und behalten oder verwerfen
- Legacy-Metrik-Kompatibilitaet: `val_bpb`

### Fine-Tune-Runtime

- Source of Truth: `finetune/trainer.py`
- Backends: `peft_trl` als stabiler Pfad, `unsloth` als optionaler Benchmark-Pfad
- oeffentliche Config-Oberflaeche: Backend, Dataset-Format, Metrik-Richtung, rsLoRA-Flag, Release-Metriken

### Release- und Publishing-Runtime

- Source of Truth: `finetune/model_publisher.py`
- Outputs: Model Card, Training Manifest, Benchmark Summary, Environment Manifest, Tester Attestation, Validation Result, License Manifest, optionales `Modelfile`
- oeffentliche Publikationsziele: GitHub Releases, Hugging Face, Ollama-kompatibles GGUF-Packaging

### Validation- und Proof-Layer

- Source of Truth: `validation/registry.json`
- Zweck: oeffentliche Hardware-Claims und Modell-Promotions gatekeeping
- Proof-Regel: zwei unabhaengige erfolgreiche End-to-End-Runs pro oeffentlichem Hardware-Tier

### Audit- und Governance-Layer

- Source of Truth: [LOGGING_AUDIT_PROTOCOL-DE.md](LOGGING_AUDIT_PROTOCOL-DE.md), [COMPLIANCE_PACK-DE.md](COMPLIANCE_PACK-DE.md), [RELEASE_POLICY-DE.md](RELEASE_POLICY-DE.md)
- Zweck: das Repo audit-ready, attribution-sicher und evidence-bound zu halten

## Oeffentliche Interfaces

- CLI: `python -m finetune.trainer`
- Release-Bundle-Manifeste: `training-manifest.json`, `benchmark-summary.json`, `environment-manifest.json`, `tester-attestation.json`, `validation-result.json`
- Maschinenlog: `results/protocol.jsonl`
- Registry: `validation/registry.json`

## Wissensgrenzen

- private Source of Truth: open-notebook
- oeffentliche Exporte: Dokus, Templates, Benchmark-Evidence, Release-Bundles, Blog-faehige Inhalte
- referenziertes Subsystem fuer Legal-Source-Provenance: [LEGAL_SOURCE_REFERENCES-DE.md](LEGAL_SOURCE_REFERENCES-DE.md)

## Betriebsregeln

- ausgelieferte Controls bleiben Default, bis ein Kandidat auf gleicher Metrik und gleichem Hardware-Budget gewinnt
- die 3090 ist das oeffentliche Referenz-Tier fuer v1
- staerkere Hardware wird unterstuetzt, aber Claims muessen hardware-spezifisch bleiben
- TuneForge bleibt im Preview-Status, bis oeffentliche Evidence die Validation Matrix belegt
