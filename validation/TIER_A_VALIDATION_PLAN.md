# Tier A Validation Run Plan (RTX 3090)

**Status:** Ready for Execution  
**Target Hardware:** NVIDIA RTX 3090 (24 GB VRAM)  
**Goal:** Verify TuneForge v1.0 on Tier A hardware and update validation registry

---

## Overview

This document provides the complete execution plan for the Tier A hardware validation run. This is a **quality gate** - successful completion moves TuneForge from "Technical Preview" to "Verified on RTX 3090".

### Success Criteria

| # | Criterion | Verification Method |
|---|-----------|---------------------|
| 1 | Complete training run (Docker → Export) | End-to-end test passes |
| 2 | Deterministic results (2 runs within ±2%) | Compare run #1 vs run #2 |
| 3 | All artifacts generated | Checklist verification |
| 4 | Registry updated with evidence | validation/registry.json updated |
| 5 | No critical errors | Log analysis |

---

## Pre-Flight Checklist

### Hardware Requirements

- [ ] NVIDIA RTX 3090 GPU (or equivalent 24GB VRAM)
- [ ] CUDA 12.4+ installed and functional
- [ ] NVIDIA Container Toolkit (nvidia-docker2) installed
- [ ] Minimum 50 GB free disk space
- [ ] Internet connectivity (for model download)

### Software Prerequisites

```bash
# Verify CUDA
nvidia-smi
# Expected: CUDA 12.4+, Driver 535+

# Verify Docker with GPU support
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Verify TuneForge version
git log --oneline -1
# Expected: Latest commit from main branch
```

### Environment Setup

```bash
# Set environment variables
export HF_TOKEN="your_huggingface_token"  # Optional: for HF Hub download caching
export WANDB_DISABLED="true"  # Disable W&B for reproducibility

# Clean up previous runs (if any)
docker system prune -f
rm -rf ./output/validation-run-*
```

---

## Test Configuration

### Config File: `configs/tier-a-validation.yaml`

```yaml
base_model: "Qwen/Qwen2.5-0.5B-Instruct"
output_dir: "output/validation-run-{timestamp}"
backend: "peft_trl"

dataset_path: "data_utils/dummy_test.jsonl"
dataset_format: "alpaca"
eval_split_ratio: 0.1

primary_metric: "eval_loss"
metric_goal: "minimize"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
use_rslora: false
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

bits: 4
bnb_4bit_compute_dtype: "bfloat16"

learning_rate: 2e-4
num_train_epochs: 1
max_steps: 10  # Minimal for validation
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
max_seq_length: 512

logging_steps: 1
save_steps: 5
eval_steps: 5

seed: 42
```

### Why This Configuration?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `base_model` | 0.5B | Fast download, fits in any VRAM |
| `max_steps` | 10 | Quick validation (not full training) |
| `max_seq_length` | 512 | Standard use case |
| `lora_r` | 16 | Balanced quality/speed |
| `seed` | 42 | Reproducibility |

---

## Execution Procedure

### Step 1: Run #1 (Primary)

```bash
# Create timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="output/validation-run-${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Start the run
docker compose -f docker-compose.finetune.yml down 2>/dev/null || true

AUTORESEARCH_DOMAIN=tier-a-validation \
  docker compose -f docker-compose.finetune.yml up --build 2>&1 | \
  tee "${RUN_DIR}/run-1.log"

# Record completion
RUN1_DIR=$(ls -td output/validation-run-* | head -1)
echo "Run 1 completed: ${RUN1_DIR}" | tee -a "${RUN_DIR}/run-1.log"
```

**Expected Duration:** 5-15 minutes (depending on download speed)

### Step 2: Run #2 (Verification)

```bash
# Wait 30 seconds between runs
sleep 30

# Clean Docker cache to ensure fresh environment
docker compose -f docker-compose.finetune.yml down
docker system prune -f

# Run #2
AUTORESEARCH_DOMAIN=tier-a-validation \
  docker compose -f docker-compose.finetune.yml up --build 2>&1 | \
  tee "${RUN_DIR}/run-2.log"

RUN2_DIR=$(ls -td output/validation-run-* | head -1)
echo "Run 2 completed: ${RUN2_DIR}" | tee -a "${RUN_DIR}/run-2.log"
```

### Step 3: Evidence Collection

```bash
#!/bin/bash
# collect-evidence.sh

RUN1_DIR="$1"
RUN2_DIR="$2"
OUTPUT_DIR="$3"

mkdir -p "${OUTPUT_DIR}"

# Copy logs
cp "${RUN1_DIR}/run-1.log" "${OUTPUT_DIR}/"
cp "${RUN2_DIR}/run-2.log" "${OUTPUT_DIR}/"

# Extract metrics
python3 << 'PYTHON'
import json
import re
import sys

def extract_metrics(log_file):
    metrics = {}
    with open(log_file) as f:
        content = f.read()
    
    # Look for primary_metric_value pattern
    match = re.search(r'primary_metric_value:\s+([\d.]+)', content)
    if match:
        metrics['primary_metric_value'] = float(match.group(1))
    
    # Look for eval_loss
    match = re.search(r'eval_loss:\s+([\d.]+)', content)
    if match:
        metrics['eval_loss'] = float(match.group(1))
    
    # Look for peak_vram_mb
    match = re.search(r'peak_vram_mb:\s+([\d.]+)', content)
    if match:
        metrics['peak_vram_mb'] = float(match.group(1))
    
    # Look for training_seconds
    match = re.search(r'training_seconds:\s+([\d.]+)', content)
    if match:
        metrics['training_seconds'] = float(match.group(1))
    
    return metrics

run1 = extract_metrics(sys.argv[1])
run2 = extract_metrics(sys.argv[2])

comparison = {
    'run1': run1,
    'run2': run2,
    'difference_percent': {
        k: abs(run1.get(k, 0) - run2.get(k, 0)) / max(run1.get(k, 1), 0.001) * 100
        for k in set(run1.keys()) & set(run2.keys())
    },
    'within_tolerance': all(
        abs(run1.get(k, 0) - run2.get(k, 0)) / max(run1.get(k, 1), 0.001) < 0.02
        for k in set(run1.keys()) & set(run2.keys())
    )
}

print(json.dumps(comparison, indent=2))
PYTHON
"${OUTPUT_DIR}/run-1.log" "${OUTPUT_DIR}/run-2.log" > "${OUTPUT_DIR}/comparison.json"

# Copy bundle artifacts
find "${RUN1_DIR}" -name "training-manifest.json" -exec cp {} "${OUTPUT_DIR}/" \;
find "${RUN1_DIR}" -name "benchmark-summary.json" -exec cp {} "${OUTPUT_DIR}/" \;
find "${RUN1_DIR}" -name "model-card.md" -exec cp {} "${OUTPUT_DIR}/" \;

echo "Evidence collected in: ${OUTPUT_DIR}"
```

---

## Verification Checklist

### After Run #1

- [ ] Docker container started successfully
- [ ] Base model downloaded (`Qwen/Qwen2.5-0.5B-Instruct`)
- [ ] Training completed (10 steps)
- [ ] Evaluation executed
- [ ] Model artifacts saved
- [ ] Release bundle generated
- [ ] No errors in logs (grep -i "error\|exception\|traceback")

### After Run #2

- [ ] Docker container started successfully
- [ ] Training completed (10 steps)
- [ ] Results within ±2% of Run #1

### Evidence Package Contents

```
validation-evidence/
├── run-1.log
├── run-2.log
├── comparison.json
├── training-manifest.json
├── benchmark-summary.json
├── model-card.md
├── hardware-specs.txt
└── TESTER_ATTESTATION.md
```

---

## Registry Update

After successful validation, update `validation/registry.json`:

```json
{
  "public_status": "technical_preview",
  "runs": [
    {
      "tier": "tier_a_rtx_3090_24gb",
      "date": "2026-04-XX",
      "tester": "[Tester ID]",
      "result": "success",
      "evidence_path": "validation/evidence/2026-04-XX-run-1",
      "git_sha": "[Git SHA]",
      "config": "configs/tier-a-validation.yaml"
    },
    {
      "tier": "tier_a_rtx_3090_24gb",
      "date": "2026-04-XX",
      "tester": "[Tester ID]",
      "result": "success",
      "evidence_path": "validation/evidence/2026-04-XX-run-2",
      "git_sha": "[Git SHA]",
      "config": "configs/tier-a-validation.yaml"
    }
  ],
  "tiers": {
    "tier_a_rtx_3090_24gb": {
      "label": "Verified on RTX 3090",
      "notes": "Primary public reference tier for TuneForge v1.0.",
      "required_successful_runs": 2,
      "status": "verified"
    },
    "tier_b_48gb_plus": {
      "label": "Verified on 48 GB+",
      "notes": "Secondary enterprise-grade validation tier for TuneForge v1.0.",
      "required_successful_runs": 2,
      "status": "unverified"
    }
  },
  "version": 1
}
```

---

## Troubleshooting

### Common Issues

#### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `max_seq_length` to 256 or `per_device_train_batch_size` to 1

#### Issue: Model Download Fails
```
Error: Connection timeout to HuggingFace
```
**Solution:** Set `HF_TOKEN` and retry, or use offline mode with cached model

#### Issue: Docker GPU Access
```
docker: Error response from daemon: could not select device driver
```
**Solution:** Install NVIDIA Container Toolkit:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Issue: Results Not Within ±2%
**Solution:** 
1. Check that `seed=42` is set in config
2. Verify no other processes using GPU
3. Check temperature throttling (nvidia-smi)
4. Increase tolerance temporarily if caused by minor numerical differences

---

## Sign-Off

### Pre-Run Checklist (Before Starting)
- [ ] Hardware verified (RTX 3090, 24GB)
- [ ] Docker + nvidia-docker2 installed
- [ ] Internet connectivity confirmed
- [ ] 50GB disk space available
- [ ] Git SHA recorded: `________________`

### Post-Run Sign-Off
- [ ] Run #1 completed successfully
- [ ] Run #2 completed successfully
- [ ] Results within ±2% tolerance
- [ ] Evidence package complete
- [ ] Registry updated
- [ ] PR created with evidence

### Tester Information
```
Tester Name: ________________
Organization: ________________
Date: ________________
Signature: ________________
```

---

*This plan must be followed exactly. Any deviations must be documented with justification.*
