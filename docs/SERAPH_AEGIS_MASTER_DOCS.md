# Seraph Aegis & TuneForge: Master System Documentation

This document serves as the authoritative technical reference for the "Zeroth-Law" hardening implementation.

## 1. System Philosophy: The Zeroth Law
The system is built on the **Zeroth Law of AI Safety**: *An AI system may not harm humanity, or, through inaction, allow humanity to come to harm.* 

To enforce this, we have moved beyond simple scripts to a **Distributed Security Architecture** where safety is not an "option" but a hard, gated constraint enforced by a high-integrity core.

---

## 2. Architecture Overview

### Component Diagram
- **Seraph Aegis (Rust)**: Central Security Authority.
- **TuneForge (Python)**: Training Node (Worker).
- **Communication**: JWT-authenticated REST (Fail-Closed).

---

## 3. Seraph Aegis (Rust Backend)

Seraph Aegis is the "Incorruptible Core" of the system, written in Rust for high performance and memory safety.

### 🛡️ Security & Auth
- **Auth-Default**: Every endpoint requires a valid `HS256` JWT.
- **Fail-Closed**: Any failure in the security stack results in a hard `DENY`.
- **Scopes**: Fine-grained control via `train`, `publish`, `admin`, and `read_policy` scopes.

### 📜 Policy Engine
- **Strategy**: Deny-Wins. Unknown tags are rejected by default.
- **Constitutional Rules**:
    - `ZL-DENY-001`: Blocks training for weapons/malware/explosives.
    - `ZL-ALLOW-001`: Explicitly permits industrial (SPS) and documentation tasks.
- **Clearance Tokens**: Generates an HMAC-signed digital receipt for every approved job, ensuring a non-repudiable audit trail.

### 📊 Service Metrics
Exposed via `GET /metrics` in Prometheus format:
- `aegis_ready`: Gauge (0=Starting, 1=Ready, 2=Degraded, 3=Failed).
- `aegis_requests_total`: Counter by method/path/status.
- `aegis_latency_ms`: Histogram for SLI monitoring.

---

## 4. TuneForge Hardening (Python)

### 🩺 SafeGrad: Adaptive Gradient Surgery
The `SafeQLoRATrainer` (in `safe_trainer.py`) implements **Orthogonal Gradient Projection**. 

**The Formula**:
$$ g_{task}' = g_{task} - \frac{g_{task} \cdot g_{safe}}{||g_{safe}||^2} \cdot g_{safe} $$

This ensures that the weight updates for a specific task never move into the direction that would degrade the safety constraints established by the "Golden Safety Dataset".

**Robustness Features**:
- **Anti-OOM**: Safety passes are forced to `batch_size=1` with automatic CUDA cache purging.
- **NaN-Guard**: Layer-wise finite-checks prevent gradient explosions.
- **Thresholding**: Intervention only occurs when safety loss $\mathcal{L}_{safe} > 0.5$, protecting model utility.

### 📉 Operability Taxonomy
The `StateMachine` in `operability.py` manages the training lifecycle:
- `OPERATIONAL`: Healthy.
- `DEGRADED_VRAM`: Running in micro-batch safety mode.
- `DEGRADED_NAN`: Layer-wise quarantine active.
- `HALTED_AEGIS_DENY`: Security core blocked the job.

---

## 5. Evaluation Harness (Red-Teaming)

Located in `eval/red_team_harness.py`, this is the final quality gate.

- **Adversarial Probes**: 8+ categories including Jailbreaks, Weapons, and Bioweapons.
- **Refusal Classification**: Heuristic-based classification of model responses.
- **Pass Threshold**: Requires **90% pass rate** for model approval.
- **Gating**: Returns Exit Code `2` if a model fails, signaling the publisher to block distribution.

---

## 6. Audit & Compliance

Every intervention is logged in two places:
1.  **Local Node**: `zeroth_audit.jsonl` (step-by-step telemetry).
2.  **Central Core**: Structured JSON access logs with UUID correlation.

---

## 7. Operational Manual

### Running the Security Core
```bash
cd zeroth
cargo run --release
```

### Starting a Hardened Training Job
Ensure the environment is configured:
```powershell
$env:AEGIS_JWT_TOKEN = "your-signed-token"
$env:AEGIS_API_URL = "http://localhost:8741"
python scripts/train_safe.py --config config.yaml
```

### Verifying System Health
```bash
curl http://localhost:8741/v1/status -H "Authorization: Bearer $TOKEN"
```
