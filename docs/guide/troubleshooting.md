# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### CUDA not available

**Symptom:**
```
RuntimeError: No CUDA GPUs are available
```

**Solution:**
```bash
# Verify CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Import Error: datasets

**Symptom:**
```
ImportError: cannot import name 'Dataset' from 'datasets'
```

**Cause:** Naming conflict with local `data_utils/` module.

**Solution:**
Use `data_utils` (not `datasets`) for local module:
```python
from data_utils.data_formats import normalize_records_to_text
```

## Training Issues

### CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

2. Reduce sequence length:
```yaml
max_seq_length: 256
```

3. Use gradient checkpointing:
```yaml
gradient_checkpointing: true
```

4. Reduce LoRA rank:
```yaml
lora_r: 8  # instead of 16 or 32
```

### Training is very slow

**Symptom:** Training takes hours for small datasets.

**Solutions:**

1. Use Unsloth backend:
```yaml
backend: "unsloth"
```

2. Verify GPU utilization:
```bash
watch -n 1 nvidia-smi
```

3. Reduce logging frequency:
```yaml
logging_steps: 100
```

### Loss is NaN

**Symptom:**
```
Loss: nan
```

**Solutions:**

1. Reduce learning rate:
```yaml
learning_rate: 1e-4  # instead of 2e-4
```

2. Enable gradient clipping:
```yaml
max_grad_norm: 0.3
```

3. Check for invalid data:
```python
# Verify no inf/nan in dataset
import json
for line in open("dataset.jsonl"):
    data = json.loads(line)
    assert all(isinstance(v, str) for v in data.values())
```

## Dataset Issues

### Dataset format errors

**Symptom:**
```
ValueError: Could not normalize record
```

**Solution:**

Verify dataset format matches config:

**Alpaca format:**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**ShareGPT format:**
```json
{"conversations": [
  {"from": "human", "value": "..."},
  {"from": "gpt", "value": "..."}
]}
```

### Empty dataset error

**Symptom:**
```
ValueError: Dataset is empty
```

**Solution:**

Check that JSONL file has content:
```bash
wc -l dataset.jsonl
head -5 dataset.jsonl
```

## Docker Issues

### Docker GPU access denied

**Symptom:**
```
docker: Error response from daemon: could not select device driver
```

**Solution:**

Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Container exits immediately

**Symptom:** Docker container starts then exits.

**Solution:**

Check logs:
```bash
docker compose -f docker-compose.finetune.yml logs
```

Common causes:
- Missing `AUTORESEARCH_DOMAIN` env var
- Invalid config file path
- Permission issues with output directory

## Export Issues

### Model card generation fails

**Symptom:**
```
KeyError: 'model_name'
```

**Solution:**

Ensure all required fields in model info:
```python
model_info = {
    "model_name": "my-model",
    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "domain": "code",
    "language": "en",
    # ... other required fields
}
```

### HuggingFace upload fails

**Symptom:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Solution:**

Set HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
# or
huggingface-cli login
```

## Validation Issues

### Tests fail with import errors

**Symptom:**
```
ModuleNotFoundError: No module named 'finetune'
```

**Solution:**

Install in editable mode:
```bash
pip install -e ".[dev]"
```

### Coverage is low

**Symptom:** Tests pass but coverage < 80%.

**Solution:**

Run with coverage:
```bash
pytest tests/ --cov=finetune --cov-report=term-missing
```

Add tests for uncovered lines.

## Getting Help

If your issue isn't listed:

1. Check [GitHub Issues](https://github.com/AI-Engineerings-at/tuneforge/issues)
2. Search existing discussions
3. Create a new issue with:
   - Full error message
   - Steps to reproduce
   - Environment info (`python --version`, `nvidia-smi`)
   - Minimal config to reproduce
