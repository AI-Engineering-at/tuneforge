# TuneForge GPU Tiers
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: GPU_TIERS-DE.md

Current public status: **Technical Preview**.

## Hardware Classes

| Tier | Typical Hardware | Primary Role | Public Claim Level |
|------|------------------|--------------|--------------------|
| 8 GB | RTX 2060, RTX 3060, mobile GPUs | smoke tests, autonomous loop, tiny experiments | no verified hardware label |
| 24 GB | RTX 3090, RTX 4090 | primary single-GPU fine-tune and benchmark tier | verification target for v1 |
| 48 GB+ | A6000, RTX 6000 Ada, A100, H100 | heavier benchmarks, larger batches, 14B-class evaluation | secondary verification target |

## Product Position

- TuneForge v1 is 3090-first
- stronger cards are supported as scale-up tiers
- stronger cards do not automatically justify different public claims
- public comparisons must always state exact hardware, VRAM class, and runtime context

## Model Guidance

- 7B and 8B models are the main local class for 24 GB
- 14B models belong to the 48 GB+ validation tier
- `Qwen/Qwen3-8B` is the main current benchmark candidate
- `Mistral Small 4` is not part of the runnable v1 matrix

## Release Rule

No tier may use a verified label until [VALIDATION_MATRIX-EN.md](VALIDATION_MATRIX-EN.md) and [../validation/registry.json](../validation/registry.json) prove the requirement.
