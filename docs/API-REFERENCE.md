# TuneForge API Reference

## Table of Contents

- [ZerothClient](#zerothclient)
- [AegisClient](#aegisclient)
- [QLoRAConfig](#qloraconfig)

---

## ZerothClient

Contract 3 client for weight update evaluation.

**Module:** `finetune.zeroth_client`

### Class: ZerothClient

```python
class ZerothClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        jwt_token: str | None = None,
    )
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| base_url | str or None | http://localhost:8741 | Zeroth HTTP endpoint |
| timeout_ms | int | 100 | Request timeout in milliseconds |
| jwt_token | str or None | None | JWT bearer token for authentication |

**Environment Variables:**
- ZEROTH_URL - Overrides base_url
- ZEROTH_TIMEOUT_MS - Overrides timeout_ms
- ZEROTH_JWT_TOKEN - Overrides jwt_token

### Method: evaluate_weight_update

```python
def evaluate_weight_update(
    self,
    model_id: str,
    delta_weights: dict[str, Any],
    training_config: dict[str, Any],
) -> ZerothResponse
```

Evaluates a weight update with Zeroth. Fail-closed: timeout or error raises exception.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| model_id | str | Unique model identifier |
| delta_weights | dict | Weight gradients by parameter name |
| training_config | dict | Training configuration metadata |

**Returns:** ZerothResponse - Contains decision, risk_score, reason

**Raises:**
- ZerothRejectError - If Zeroth returns DENY decision
- ZerothTimeoutError - If request times out (fail-closed)
- ZerothClientError - For connection errors (fail-closed)

**Example:**
```python
from finetune.zeroth_client import ZerothClient

client = ZerothClient(
    base_url="http://zeroth:8741",
    timeout_ms=100,
    jwt_token="eyJ0eXAiOiJKV1Qi..."
)

try:
    response = client.evaluate_weight_update(
        model_id="my-model-v1",
        delta_weights={"layer1.weight": [0.1, 0.2]},
        training_config={"lr": 0.001, "step": 100}
    )
    if response.allowed:
        print(f"Safe to proceed: {response.reason}")
except ZerothRejectError as e:
    print(f"Rejected: {e.reason} (risk: {e.risk_score})")
```

### Method: evaluate_or_raise

```python
def evaluate_or_raise(
    self,
    model_id: str,
    delta_weights: dict[str, Any],
    training_config: dict[str, Any],
) -> None
```

Convenience method that raises ZerothRejectError on DENY.

**Parameters:** Same as evaluate_weight_update

**Returns:** None (raises on rejection)

**Raises:**
- ZerothRejectError - If decision is DENY
- ZerothTimeoutError - On timeout
- ZerothClientError - On connection error

### Class: ZerothResponse

```python
@dataclass(frozen=True)
class ZerothResponse:
    decision: Decision      # Decision.ALLOW or Decision.DENY
    risk_score: float       # 0.0 to 1.0
    reason: str             # Human-readable explanation
```

**Properties:**
- allowed: bool - True if decision is ALLOW

### Exceptions

**ZerothClientError**
Base exception for all Zeroth client errors.

**ZerothTimeoutError(ZerothClientError)**
Raised when request exceeds timeout. Fail-closed behavior.

**ZerothRejectError(ZerothClientError)**
Raised when Zeroth denies the request.

**Attributes:**
- reason: str - Rejection reason
- risk_score: float - Risk assessment (0.0 - 1.0)

### Function: create_zeroth_client

```python
def create_zeroth_client(
    base_url: str | None = None,
    timeout_ms: int | None = None,
) -> ZerothClient
```

Factory function that creates client from environment variables.

**Environment Variables:**
- ZEROTH_URL - Base URL
- ZEROTH_TIMEOUT_MS - Timeout
- ZEROTH_JWT_TOKEN - JWT token

**Returns:** Configured ZerothClient instance

---

## AegisClient

Pre/post-training safety validation client.

**Module:** `finetune.operability`

### Class: AegisClient

```python
class AegisClient:
    def __init__(self)
```

Reads configuration from environment variables:
- AEGIS_API_URL - Default: http://localhost:8741
- AEGIS_JWT_TOKEN - JWT bearer token
- AEGIS_TIMEOUT_SEC - Default: 5
- ZEROTH_MOCK_MODE - Set to 1 for development (blocks in production)

**Production Safety:**
If ZEROTH_MOCK_MODE=1 and TUNEFORGE_ENV=production, calls sys.exit(1).

### Method: evaluate_policy

```python
def evaluate_policy(
    self,
    job_id: str,
    tags: list[str],
    phase: str,           # "pre_train" or "pre_publish"
    dataset_hash: str,
    node_id: Optional[str] = None,
) -> dict
```

Calls POST /v1/policy/evaluate on Seraph Aegis.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| job_id | str | Unique job identifier |
| tags | list[str] | Safety tags (e.g., ["medical", "eu"]) |
| phase | str | "pre_train" or "pre_publish" |
| dataset_hash | str | SHA-256 hash of training data |
| node_id | Optional[str] | Node identifier (default: env TUNEFORGE_NODE_ID) |

**Returns:**
```python
{
    "verdict": "ALLOW" or "DENY",
    "reason": str,
    "clearance_token": str,
    "verification_status": "VERIFIED" or "UNVERIFIED"  # Mock mode
}
```

**Raises:**
- RuntimeError - On denial or connection error (if not mock mode)

---

## QLoRAConfig

Training configuration dataclass.

**Module:** `finetune.trainer`

### Class: QLoRAConfig

```python
@dataclass
class QLoRAConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "output"
    backend: str = "peft_trl"  # or "unsloth"
    trust_remote_code: bool = False  # Security: opt-in only
    
    # Dataset
    dataset_path: str = ""
    dataset_format: str = "alpaca"  # or "sharegpt", "text"
    eval_split_ratio: float = 0.1
    
    # Metrics
    primary_metric: str = "eval_loss"
    metric_goal: str = "minimize"  # or "maximize"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    use_rslora: bool = False
    target_modules: list[str]  # Default: q_proj, k_proj, v_proj, etc.
    
    # Training
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42
```

### Method: from_yaml

```python
@staticmethod
def from_yaml(path: str) -> QLoRAConfig
```

Load configuration from YAML file.

**Parameters:**
- path: str - Path to YAML file

**Returns:** QLoRAConfig instance

**Raises:**
- FileNotFoundError - If file does not exist
- ValueError - If YAML is invalid

**Example:**
```python
config = QLoRAConfig.from_yaml("configs/training.yaml")
```

### Method: to_yaml

```python
def to_yaml(self, path: str) -> None
```

Save configuration to YAML file.

**Parameters:**
- path: str - Output file path

### Method: to_dict

```python
def to_dict(self) -> dict[str, Any]
```

Convert configuration to dictionary.

**Returns:** Dictionary representation

---

## Environment Variables Reference

### Zeroth Integration

| Variable | Default | Description |
|----------|---------|-------------|
| ZEROTH_URL | http://localhost:8741 | Zeroth HTTP endpoint |
| ZEROTH_TIMEOUT_MS | 100 | Contract 3 timeout |
| ZEROTH_JWT_TOKEN | None | Authentication token |
| ZEROTH_MOCK_MODE | 0 | Development bypass (blocks in production) |

### Aegis Integration

| Variable | Default | Description |
|----------|---------|-------------|
| AEGIS_API_URL | http://localhost:8741 | Aegis HTTP endpoint |
| AEGIS_JWT_TOKEN | None | Authentication token |
| AEGIS_TIMEOUT_SEC | 5 | Policy evaluation timeout |

### Training Environment

| Variable | Default | Description |
|----------|---------|-------------|
| TUNEFORGE_ENV | development | Environment name (production blocks mock mode) |
| TUNEFORGE_NODE_ID | unknown | Node identifier for distributed training |
| HF_TOKEN | None | HuggingFace authentication |

---

## HTTP Endpoints (Zeroth Service)

### POST /v1/policy/evaluate

Pre-training and pre-publish safety checks.

**Request:**
```json
{
  "node_id": "worker-1",
  "job_id": "train-abc123",
  "tags": ["medical", "eu"],
  "phase": "pre_train",
  "dataset_hash": "sha256:abc..."
}
```

**Response:**
```json
{
  "verdict": "ALLOW",
  "reason": "Dataset verified safe",
  "clearance_token": "tok_abc123",
  "verification_status": "VERIFIED"
}
```

### POST /evaluate

Contract 3 - Weight update evaluation (mid-training).

**Request:**
```json
{
  "action": "weight_update",
  "model_id": "my-model",
  "delta_weights_hash": "sha256:abc...",
  "training_config": {
    "learning_rate": 0.001,
    "global_step": 100
  }
}
```

**Response:**
```json
{
  "decision": "allow",
  "risk_score": 0.1,
  "reason": "Update within safe bounds"
}
```

**Headers:**
- Content-Type: application/json
- Authorization: Bearer <JWT_TOKEN>
