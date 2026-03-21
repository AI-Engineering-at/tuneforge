#!/bin/bash
set -e

DOMAIN=${AUTORESEARCH_DOMAIN:-sps-plc}
CONFIG="/app/finetune/configs/${DOMAIN}.yaml"

echo "=== TuneForge Studio Fine-Tuning ==="
echo "Domain: ${DOMAIN}"
echo "Config: ${CONFIG}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs:"
    ls /app/finetune/configs/
    exit 1
fi

DATA_DIR="/app/datasets/generated/${DOMAIN//-/}"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "No training data found in $DATA_DIR"
    echo "Generate data first: python -m datasets.synthetic_generator --domain $DOMAIN"
    exec /bin/bash
fi

echo "Training data: $(wc -l ${DATA_DIR}/*.jsonl | tail -1) examples"
echo ""
echo "Ready. Run: python -m finetune.trainer --config $CONFIG"
exec /bin/bash
