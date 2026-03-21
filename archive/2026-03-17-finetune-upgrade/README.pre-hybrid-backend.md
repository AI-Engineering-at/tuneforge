# autoresearch-kit — AI Model Training on Consumer GPUs

[![Fork of karpathy/autoresearch](https://img.shields.io/badge/fork-karpathy%2Fautoresearch-blue)](https://github.com/karpathy/autoresearch)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://docs.docker.com/get-docker/)
[![GPU Tiers](https://img.shields.io/badge/GPU-6--24GB%20VRAM-green)](#gpu-tiers)
[![License](https://img.shields.io/badge/license-MIT%20%28upstream%29-lightgrey)](upstream/LICENSE)

> Train specialized AI models overnight on your existing hardware. No cloud. No API costs.

This kit packages [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for consumer NVIDIA GPUs (6–24GB VRAM). The original targets H100s (80GB). We adapt it for the hardware most engineers and companies actually own.

---

## What It Does

autoresearch runs an autonomous loop: an AI agent modifies the training code, runs a 5-minute experiment, measures the result, and keeps only what improves the model. Repeat overnight.

Karpathy's kit proved this works on research hardware. This kit makes it work on yours.

**In one sentence:** Give it a task, give it your GPU, let it run. Wake up to a specialized model that fits your use case — at zero marginal inference cost.

---

## Key Results (RTX 4070 Laptop, 8GB VRAM — measured 2026-03-12)

| Config | Params | Best val_bpb | VRAM | Throughput |
|--------|--------|--------------|------|------------|
| Small (depth 3, 384 dim) | 11.6M | **0.597** | 2.3 GB | 135K tok/sec |
| Medium (depth 6, 512 dim) | 24.6M | 0.607 | 2.3 GB | 68K tok/sec |
| Large (depth 8, 768 dim) | 64.5M | 0.657 | 2.6 GB | 32K tok/sec |

**What this means:**
- 15 automated experiments ran without manual intervention
- The agent found the optimal learning rate (0.001 → 0.0005) on its own
- VRAM usage stays well below the 8GB limit — no OOM crashes
- Throughput at 135K tok/sec means ~40M tokens per 5-minute run

> Note: All benchmarks use TinyStories (simple children's text). Results are not comparable
> to ClimbMix benchmarks from the original project. Tier 2/3 (ClimbMix) not yet tested.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    autoresearch Loop                        │
│                                                             │
│   Agent (Claude Code / GPT-4 / local LLM)                  │
│       ↓ reads program.md (research instructions)            │
│       ↓ modifies train.py (architecture / optimizer / LR)   │
│       ↓ runs training for exactly 5 minutes                 │
│       ↓ reads val_bpb from results.tsv                      │
│       ↓ better? → keep  |  worse? → revert                  │
│       ↓ repeat                                              │
│                                                             │
│   Output: improved train.py + results.tsv + trained model   │
└─────────────────────────────────────────────────────────────┘
```

The loop is fully autonomous. You point your AI agent at `programs/program-conservative.md`
and let it iterate. No human in the loop required between experiments.

---

## Programs (Research Strategies)

Programs are instruction files for the AI agent. Each defines a different research approach.

| Program | File | Strategy |
|---------|------|----------|
| **Default** | `programs/program-default.md` | Karpathy's original — broad exploration |
| **Conservative** | `programs/program-conservative.md` | One change at a time, aggressive reverting |
| **Hyperparams** | `programs/program-hyperparams.md` | Focus on learning rates, batch sizes, warmup |
| **Architecture** | `programs/program-architecture.md` | Depth, width, attention heads |

**Recommendation for 8GB GPUs:** Start with `program-conservative.md`. Our benchmark found
that LR changes deliver the highest ROI for 5-minute runs. Architecture changes need longer
experiments to show benefit.

---

## Documentation

- [Setup Guide (English)](docs/SETUP-EN.md)
- [Setup Guide (Deutsch)](docs/SETUP-DE.md)
- [GPU Tiers](docs/GPU-TIERS.md)
- [Use Cases](docs/USE_CASES.md)
- [Business ROI](docs/BUSINESS_ROI.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Fork Attribution](FORK.md)
- [Credits](docs/CREDITS.md)

---

## Credits & Attribution

This kit is built on top of **[autoresearch](https://github.com/karpathy/autoresearch)** by
[Andrej Karpathy](https://karpathy.ai), released under the MIT License.

The original code is in `upstream/` — unchanged. Our consumer GPU adaptations
are separate files in `patches/`, `configs/`, and root. Attribution stays clear.

- Original repository: https://github.com/karpathy/autoresearch
- Original license: [MIT](upstream/LICENSE)
- Announcement: https://x.com/karpathy/status/2029701092347630069
