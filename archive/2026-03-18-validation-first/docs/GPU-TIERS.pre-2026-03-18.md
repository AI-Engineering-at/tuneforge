# GPU Tiers

## Product Tiers

TuneForge separates research and fine-tune expectations by hardware tier.

| Tier | Hardware | Expected Scope |
|------|----------|----------------|
| 8 GB | RTX 2060, 3060, 4060, laptop GPUs | autonomous loop, TinyStories-scale experiments, smoke benchmarking |
| 24 GB | RTX 3090, RTX 4090 | primary single-GPU benchmark and fine-tune tier |
| 48 GB+ | RTX 6000 Ada, A6000, A100, H100 | larger batch budgets, stronger enterprise benchmarking, heavier candidates |

## Policy

- 24 GB remains the benchmark reference for public candidate promotion
- stronger GPUs are supported, but public claims must still state the exact hardware
- larger hardware does not justify changing the shipped control without benchmark evidence

## Candidate Guidance

- `Qwen/Qwen3-8B` is a benchmark candidate for 24 GB+
- `Mistral Small 4` is not a practical single-3090 target
- `Ministral 3 8B` remains a later text-only candidate, not the current default
