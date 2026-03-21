# TuneForge GPU-Tiers
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: GPU_TIERS-EN.md

Der aktuelle oeffentliche Status ist **Technical Preview**.

## Hardware-Klassen

| Tier | Typische Hardware | Primaere Rolle | Oeffentliches Claim-Niveau |
|------|-------------------|----------------|----------------------------|
| 8 GB | RTX 2060, RTX 3060, mobile GPUs | Smoke-Tests, autonomer Loop, kleine Experimente | kein verifiziertes Hardware-Label |
| 24 GB | RTX 3090, RTX 4090 | primaeres Single-GPU-Fine-Tune- und Benchmark-Tier | Verifikationsziel fuer v1 |
| 48 GB+ | A6000, RTX 6000 Ada, A100, H100 | schwerere Benchmarks, groessere Batches, 14B-Evaluation | sekundaeres Verifikationsziel |

## Produktposition

- TuneForge v1 ist 3090-first
- staerkere Karten sind als Scale-up-Tiers unterstuetzt
- staerkere Karten rechtfertigen nicht automatisch andere oeffentliche Claims
- oeffentliche Vergleiche muessen immer exakte Hardware, VRAM-Klasse und Laufzeitkontext nennen

## Modell-Leitplanken

- 7B- und 8B-Modelle sind die Hauptklasse fuer 24 GB
- 14B-Modelle gehoeren in das 48-GB+-Validation-Tier
- `Qwen/Qwen3-8B` ist der aktuelle Hauptkandidat
- `Mistral Small 4` ist kein Teil der lauffaehigen v1-Matrix

## Release-Regel

Kein Tier darf ein verifiziertes Label tragen, bevor [VALIDATION_MATRIX-DE.md](VALIDATION_MATRIX-DE.md) und [../validation/registry.json](../validation/registry.json) die Anforderung belegen.
