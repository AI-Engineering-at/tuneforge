# TuneForge Validation Matrix
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: VALIDATION_MATRIX-EN.md

Der aktuelle oeffentliche Status ist **Technical Preview**.

## Oeffentliche Hardware-Tiers

| Tier | Label | Umfang | Verifikationsregel |
|------|-------|--------|--------------------|
| Tier A | RTX 3090 / 24 GB | primaeres oeffentliches Fine-Tune- und Benchmark-Tier | 2 unabhaengige erfolgreiche End-to-End-Runs |
| Tier B | 48 GB+ | sekundaeres enterprise-nahes Validation-Tier | 2 unabhaengige erfolgreiche End-to-End-Runs |

## End-to-End-Proof-Anforderung

Jeder qualifizierende Run muss belegen:

- Docker-Start
- Model-Download
- Fine-Tune-Start
- Evaluation
- maschinenlesbare Metriken
- Release-Bundle-Erzeugung
- Release-Bundle-Validierung
- Hugging-Face-Publish-Dry-Run oder freigegebene echte Publikation
- Ollama-Package-Dry-Run, wenn GGUF-Packaging Teil des Runs ist

## Modellumfang pro Tier

- Tier A validiert die ausgelieferten Controls und `Qwen/Qwen3-8B`
- Tier B validiert denselben 8B-Pfad plus einen 14B-Kandidaten
- `Mistral Small 4` bleibt fuer v1 nur dokumentiert und nicht lauffaehiger Teil der Matrix

## Claim-Regeln

- oeffentliche Dokus duerfen immer `Technical Preview` sagen
- kein oeffentliches Dokument darf ein verifiziertes Hardware-Label benutzen, bevor die Registry verifiziert ist
- unveroeffentlichte, fehlgeschlagene oder hardware-unfaire Runs zaehlen nicht als oeffentliche Benchmark-Evidence
- Kandidaten duerfen nur auf gleicher Metrik und gleichem Hardware-Budget gegen die Control promoted werden

## Evidence Registry

Die Source of Truth ist [../validation/registry.json](../validation/registry.json).

Jeder akzeptierte Run muss festhalten:

- Hardware-Tier
- Modell und Config
- Run-Datum
- Tester-Identifier
- Ergebnisstatus
- Flag fuer unabhaengige Umgebung
- Artefaktpfad oder Evidence-Referenz

## Private Pilot

Externer Hardware-Proof wird ueber den Private Pilot gesammelt:

- [../validation/PRIVATE_PILOT_RUNBOOK-DE.md](../validation/PRIVATE_PILOT_RUNBOOK-DE.md)
- `validation/templates/environment-manifest.example.json`
- `validation/templates/tester-attestation.example.json`
- `validation/templates/failure-report-template.md`
