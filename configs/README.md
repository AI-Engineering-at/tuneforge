# AutoResearch Kit — Tier Configs

Configuration files for GPU-tiered training runs.

## Config Overview

| Config | GPU Range | Dataset | Model Params | device_batch_size | Description |
|--------|-----------|---------|-------------|-------------------|-------------|
| `tier1-8gb.json` | 6-8 GB | TinyStories | depth=3, embd=384, head=4 | 16 | Entry tier. Small model, 5-min experiments. LR tuned via 15 experiments. |
| `tier2-12gb.json` | 10-12 GB | ClimbMix | depth=5, embd=512, head=6 | 32 | Mid tier. ClimbMix subset, medium model. |
| `tier3-24gb.json` | 16-24 GB | ClimbMix | depth=8, embd=768, head=6 | 8 | Premium tier. Full ClimbMix, close to original autoresearch. |
| `tier3-tinystories-baseline.json` | 16-24 GB | TinyStories | depth=8, embd=768, head=6 | 8 | Baseline for quick validation runs on 24GB GPUs. |

## Tested VRAM Usage

- **tier3-24gb** with `device_batch_size=8` on RTX 4070 (12 GB): **16.6 GB VRAM peak** (BF16)
- **tier3-24gb** with `device_batch_size=8` on RTX 2060 (6 GB): runs in FP32 fallback (Turing arch, no BF16)
- `device_batch_size=64` causes OOM on 24 GB GPUs — do not use for tier3.

## Fields

| Field | Description |
|-------|-------------|
| `tier` | Tier identifier, used for auto-selection by VRAM detection |
| `name` | Display name for logs and reports (must match `tier`) |
| `min_vram_mb` / `max_vram_mb` | VRAM range for auto-tier selection |
| `dataset` | Training dataset (`tinystories` or `climbmix`) |
| `num_shards` | Number of data shards (4 for tinystories, 32-64 for climbmix) |
| `device_batch_size` | Per-GPU batch size — keep small to avoid OOM |
| `total_batch_size` | Effective batch size (gradient accumulation fills the gap) |
| `depth` / `n_head` / `n_embd` | Transformer architecture parameters |
| `window_pattern` | Attention window pattern (`L`=local, `S`=sliding, combinations) |
| `disable_compile` | Set `true` for GPUs where `torch.compile` fails (e.g., older CUDA) |

## Adding a New Config

1. Copy the closest existing tier config
2. Set `tier` and `name` to the same unique identifier
3. Adjust `min_vram_mb` / `max_vram_mb` to avoid overlap with other tiers
4. Test with a short run before publishing
