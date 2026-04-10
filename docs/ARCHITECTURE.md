# TuneForge Architecture

## System Overview

TuneForge is a benchmark-first fine-tuning framework for local LLMs with enterprise-grade safety controls.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TuneForge v1.0.0                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   Config    │   │   Dataset   │   │   Trainer   │   │  Publisher  │     │
│  │   (YAML)    │──▶│   Utils     │──▶│  (QLoRA)    │──▶│   (HF/OL)   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│         │                  │                │                  │            │
│         ▼                  ▼                ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Zeroth Safety Layer                            │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │   │
│  │  │ Pre-Train   │   │  Contract 3 │   │    Pre-Publish          │   │   │
│  │  │   Check     │   │  (Mid-Train)│   │       Check             │   │   │
│  │  │  :8741/v1   │   │  :8741/eval │   │     :8741/v1            │   │   │
│  │  └─────────────┘   └─────────────┘   └─────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System (`finetune/trainer.py`)

**QLoRAConfig** - Central configuration dataclass:
```python
@dataclass
class QLoRAConfig:
    base_model: str                    # HuggingFace model ID
    output_dir: str                    # Training output directory
    backend: str                       # "peft_trl" or "unsloth"
    trust_remote_code: bool = False    # Security: opt-in only
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str]          # Which layers to train
```

**Features:**
- YAML serialization/deserialization
- Post-init validation
- Metric goal enforcement (minimize/maximize)

### 2. Data Pipeline (`data_utils/`)

**Flow:**
```
Raw Dataset → Format Detection → Normalization → Tokenization → Batching
```

**Supported Formats:**
- Alpaca (instruction/input/output)
- ShareGPT (conversations)
- Plain text (auto-detection)

**Key Classes:**
- `data_formats.py` - Format detection and conversion
- `synthetic_generator.py` - AI-generated training data
- `legal_data.py` - Data provenance tracking

### 3. Training Engine (`finetune/trainer.py`, `finetune/safe_trainer.py`)

**Dual Backend Support:**

| Backend | Use Case | Speed | Memory |
|---------|----------|-------|--------|
| PEFT/TRL | Stability, compatibility | Baseline | Baseline |
| Unsloth | Maximum performance | +2-3x | -20% |

**SafeQLoRATrainer** (`safe_trainer.py`):
- Extends SFTTrainer with gradient surgery
- Contract 3 integration (every step evaluation)
- Safety dataset micro-batching
- Orthogonal gradient projection

### 4. Safety Layer (`finetune/zeroth_core.py`, `finetune/zeroth_client.py`)

**Three Safety Gates:**

```
Pre-Train Check          Contract 3 (Mid-Train)      Pre-Publish Check
    │                           │                            │
    ▼                           ▼                            ▼
┌─────────┐              ┌─────────────┐              ┌─────────────┐
│ Dataset │              │   Weight    │              │   Model     │
│  Hash   │              │   Update    │              │   Card      │
│  Tags   │──────────────│ Evaluation  │──────────────│  Manifest   │
│         │              │  (100ms)    │              │             │
│ :8741   │              │  :8741      │              │  :8741      │
└─────────┘              └─────────────┘              └─────────────┘
```

**Contract 3 - Weight Update Evaluation:**
- HTTP POST to `:8741/evaluate`
- Payload: `{action, model_id, delta_weights_hash, training_config}`
- Response: `{decision, risk_score, reason}`
- Timeout: 100ms
- Fail-closed: Timeout/Error → REJECT

**ZerothClient** (`zeroth_client.py`):
```python
client = ZerothClient(
    base_url="http://zeroth:8741",
    timeout_ms=100,
    jwt_token="..."  # From ZEROTH_JWT_TOKEN env
)

# Returns ZerothResponse or raises ZerothRejectError
response = client.evaluate_weight_update(...)
```

### 5. Export Pipeline (`finetune/model_publisher.py`)

**Release Bundle Contents:**
```
output/
├── adapter/              # LoRA weights
├── model_card.md         # HuggingFace model card
├── training_manifest.json
├── benchmark_summary.json
├── license_manifest.json
├── environment_manifest.json
└── tester_attestation.json
```

**Export Formats:**
- HuggingFace Hub (adapter + model card)
- GGUF ( quantized for local inference)
- Ollama (Modelfile + weights)

## Data Flow

### Training Flow

```
1. Load Config (YAML)
   ↓
2. Pre-Train Safety Check (Zeroth :8741)
   ↓
3. Load Model + Tokenizer
   ↓
4. Prepare Datasets (Train/Eval/Safety)
   ↓
5. Initialize SafeQLoRATrainer
   ↓
6. Training Loop:
   a. Forward/Backward (task data)
   b. Gradient Surgery (safety data)
   c. Contract 3 Evaluation (:8741/evaluate)
   d. Optimizer Step (if ALLOW)
   ↓
7. Export Model
   ↓
8. Pre-Publish Safety Check (Zeroth :8741)
   ↓
9. Publish to Hub
```

### Safety Evaluation Flow (Contract 3)

```
Training Step N
    │
    ▼
Compute Task Gradients
    │
    ▼
Compute Safety Gradients (orthogonal projection)
    │
    ▼
Collect Delta Weights (gradients as weight changes)
    │
    ▼
POST /evaluate (100ms timeout)
    │
    ├──► ALLOW ──► Optimizer Step
    │
    ├──► DENY ───► RuntimeError (Training Halted)
    │
    └──► TIMEOUT ─► RuntimeError (Fail-closed)
```

## Security Architecture

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  External (Untrusted)                                    │
│  - HuggingFace Hub                                      │
│  - Training datasets (user-provided)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ (validation layer)
┌─────────────────────────────────────────────────────────┐
│  Safety Layer (Zeroth)                                  │
│  - Pre-train validation                                 │
│  - Mid-train evaluation (Contract 3)                    │
│  - Pre-publish validation                               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ (only if ALLOW)
┌─────────────────────────────────────────────────────────┐
│  Core Training (Trusted)                                │
│  - SafeQLoRATrainer                                     │
│  - Gradient surgery                                     │
│  - Contract 3 client                                    │
└─────────────────────────────────────────────────────────┘
```

### Fail-Closed Design

| Scenario | Behavior |
|----------|----------|
| Zeroth timeout (100ms) | REJECT, training halted |
| Zeroth connection error | REJECT, training halted |
| Zeroth DENY response | REJECT, training halted |
| MOCK_MODE in production | `sys.exit(1)` |
| Missing JWT token | Request without auth (for dev) |

### Configuration Security

**trust_remote_code** (`trainer.py`):
- Default: `False`
- Purpose: Prevent arbitrary code execution from HF models
- Opt-in only: User must explicitly set `trust_remote_code: true`

**ZEROTH_MOCK_MODE** (`operability.py`):
- Development only
- Production detection: `TUNEFORGE_ENV=production`
- Behavior: `sys.exit(1)` with error log

## Performance Characteristics

### Contract 3 Overhead

| Metric | Value | Impact |
|--------|-------|--------|
| HTTP timeout | 100ms | Maximum latency per step |
| Payload size | ~1KB (hashed weights) | Network overhead minimal |
| Evaluation frequency | Every step | 1000 steps = ~100s max overhead |
| Fail-closed latency | Immediate | No retries, instant REJECT |

### Memory Usage

| Component | VRAM Overhead |
|-----------|---------------|
| Base model (7B QLoRA) | ~6GB |
| Gradient surgery | +200MB (safety batch_size=1) |
| Contract 3 client | Negligible (CPU-based hashing) |

## Integration Points

### Zeroth Service (:8741)

**Endpoints:**
```
POST /v1/policy/evaluate    # Pre-train, Pre-publish checks
POST /evaluate              # Contract 3 (weight updates)
```

**Authentication:**
```
Authorization: Bearer <JWT_TOKEN>
```

**Environment Variables:**
```bash
ZEROTH_URL=http://zeroth:8741
ZEROTH_TIMEOUT_MS=100
ZEROTH_JWT_TOKEN=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
ZEROTH_MOCK_MODE=0  # 1=development only, never production
```

### HuggingFace Integration

**Upload:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id=target_repo, folder_path=output_dir)
```

**Authentication:**
```bash
HF_TOKEN=hf_...  # For model upload
```

## Deployment Patterns

### Development (Local)
```bash
# Mock mode enabled for testing
ZEROTH_MOCK_MODE=1 python train.py --config config.yaml
```

### Production (Docker)
```bash
# Mock mode blocked, real Zeroth required
docker run -e TUNEFORGE_ENV=production \
           -e ZEROTH_URL=http://zeroth:8741 \
           -e ZEROTH_JWT_TOKEN=... \
           tuneforge:1.0.0
```

## Monitoring & Observability

### Structured Logging

All components use structured JSON logging:
```json
{
  "timestamp": "2026-04-10T12:00:00Z",
  "level": "INFO",
  "component": "tuneforge.zeroth_client",
  "event": "contract3_evaluation",
  "model_id": "my-model",
  "decision": "ALLOW",
  "risk_score": 0.1,
  "elapsed_ms": 45
}
```

### Audit Trail

**Zeroth Audit Log** (`output/zeroth_audit.jsonl`):
- Gradient surgery interventions
- Contract 3 evaluations
- Safety shield activations

**Validation Registry** (`validation/registry.jsonl`):
- Hardware attestation
- Tester signatures
- Verification runs

## Known Limitations

1. **Contract 3 Performance**: HTTP call on every training step adds latency
2. **Gradient Hashing**: CPU-intensive for large models (mitigated: only every step)
3. **Mock Mode Risk**: Can be accidentally enabled in production (mitigated: exit(1))
4. **JWT Token**: Must be manually rotated (no automatic refresh)

## Future Improvements

- [ ] Batch Contract 3 evaluations (every N steps with buffer)
- [ ] Async evaluation (non-blocking with callback)
- [ ] Automatic JWT token refresh
- [ ] Distributed training safety coordination
