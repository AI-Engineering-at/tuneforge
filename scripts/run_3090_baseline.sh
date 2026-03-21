#!/bin/bash
# Run the legacy autonomous baseline on a remote RTX 3090 host.
set -euo pipefail

TARGET_HOST="${1:-user@your-gpu-server}"
REMOTE_DIR="${2:-~/tuneforge}"
IMAGE_NAME="${3:-ai-engineering/tuneforge-studio:technical-preview}"

echo "=== TuneForge Technical Preview - RTX 3090 Legacy Baseline ==="
echo "Target: ${TARGET_HOST}"
echo "Mode: legacy autoresearch loop validation helper"
echo "Config: tier3-24gb (ClimbMix, 64.5M params, depth=8)"
echo ""

ssh "${TARGET_HOST}" "cd ${REMOTE_DIR} && \
    docker build -f Dockerfile -t ${IMAGE_NAME} . && \
    docker run --rm --gpus all --entrypoint bash \
        -v \$(pwd)/results:/app/results \
        -v tuneforge-cache:/root/.cache/tuneforge \
        ${IMAGE_NAME} \
        -c 'mkdir -p /app/results && \
            cp /app/patches/train_consumer.py /app/train.py && \
            cp /app/patches/prepare_consumer.py /app/prepare.py && \
            AUTORESEARCH_TIER_CONFIG=/app/configs/tier3-24gb.json \
            python3 train.py 2>&1 | tee /app/results/test-90-tier3-baseline.log'"

echo ""
echo "=== Baseline complete. Check results/test-90-tier3-baseline.log ==="
