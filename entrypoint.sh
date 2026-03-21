#!/bin/bash
set -e

echo "============================================"
echo "  TuneForge Studio"
echo "  Docker-first self-finetune runtime"
echo "  Based on github.com/karpathy/autoresearch"
echo "============================================"
echo ""

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "GPU Detected: $GPU_NAME"
echo "VRAM: ${VRAM_MB} MB"
echo ""

if [ "$VRAM_MB" -lt 9000 ]; then
    TIER="tier1-8gb"
    echo "Selected: Tier 1 (6-8GB) - TinyStories, small model"
elif [ "$VRAM_MB" -lt 15000 ]; then
    TIER="tier2-12gb"
    echo "Selected: Tier 2 (10-12GB) - ClimbMix subset, medium model"
else
    TIER="tier3-24gb"
    echo "Selected: Tier 3 (16-24GB) - ClimbMix, near-original settings"
fi

export AUTORESEARCH_TIER_CONFIG="/app/configs/${TIER}.json"
echo "Config: $AUTORESEARCH_TIER_CONFIG"
echo ""

if [ -n "$AUTORESEARCH_TIER" ]; then
    TIER="$AUTORESEARCH_TIER"
    export AUTORESEARCH_TIER_CONFIG="/app/configs/${TIER}.json"
    echo "OVERRIDE: Using tier $TIER from environment"
fi

mkdir -p /app/results

cp /app/patches/train_consumer.py /app/train.py
cp /app/patches/prepare_consumer.py /app/prepare.py

CACHE_DIR="$HOME/.cache/tuneforge"
LEGACY_CACHE_DIR="$HOME/.cache/autoresearch-consumer"
if [ ! -d "$CACHE_DIR" ] && [ -d "$LEGACY_CACHE_DIR" ]; then
    CACHE_DIR="$LEGACY_CACHE_DIR"
fi

DATA_FILES=$(find "$CACHE_DIR/data" -name "*.parquet" -o -name "*.bin" 2>/dev/null | head -1)
if [ -z "$DATA_FILES" ]; then
    echo ""
    echo "First run: downloading dataset and training tokenizer..."
    echo "This may take a few minutes..."
    python3 prepare.py --tier-config "$AUTORESEARCH_TIER_CONFIG"
    echo "Data preparation complete."
fi

echo ""
echo "============================================"
echo "  TuneForge Studio is ready."
echo ""
echo "  Quick test:  uv run train.py"
echo "  Programs:    ls /app/programs/"
echo "  Results:     /app/results/"
echo "============================================"
echo ""

exec /bin/bash
