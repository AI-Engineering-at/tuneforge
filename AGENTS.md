# TuneForge — AI Agent Guide

**Language**: EN  
**Project**: TuneForge — Benchmark-first fine-tuning framework for local LLMs  
**Status**: Technical Preview

This document provides essential context for AI coding agents working on the TuneForge codebase.

---

## 1. Project Overview

TuneForge is an open-source, audit-ready engineering framework for QLoRA fine-tuning, benchmarking, and governed model publishing. Built on the foundation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), it provides a complete pipeline from data preparation through training, evaluation, and export to Hugging Face and Ollama.

### Key Features

- **Dual Backend Training**: Switch between `transformers + peft + trl` and `unsloth` via config
- **Hardware-Tiered Configs**: Pre-tuned configurations for 8 GB, 12 GB, and 24 GB+ GPUs
- **Autonomous Agent Loop**: Provider-agnostic research loop (Claude, OpenAI, Ollama, OpenRouter, Kimi)
- **Governed Release Bundles**: Every export includes model card, training manifest, benchmark summary
- **GGUF + Ollama Export**: Convert adapters to GGUF and generate Modelfiles
- **Audit Trail**: VRAM tracking, reproducible seeds, git SHA provenance

---

## 2. Technology Stack

### Core Runtime

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.6+ with CUDA 12.4 |
| Model Training | PEFT + TRL, optional Unsloth backend |
| Quantization | BitsAndBytes 4-bit (QLoRA) |
| Tokenization | transformers, tiktoken, rustbpe |
| Data Processing | pandas, pyarrow, datasets |

### Dependencies

```toml
# Core (always required)
torch>=2.6.0, numpy, pandas, pyyaml, requests, matplotlib

# LLM Providers (optional: llm)
anthropic, openai

# Fine-tuning (optional: finetune)
accelerate, bitsandbytes, datasets, peft, transformers, trl, unsloth

# Development (optional: dev)
pytest>=8.0.0
```

### Infrastructure

- **Docker**: NVIDIA CUDA 12.4 base images
- **Container Runtime**: NVIDIA Container Toolkit (nvidia-docker2)
- **CI/CD**: GitHub Actions (workflows not committed, generated at build time)

---

## 3. Project Structure

```
tuneforge/
├── agent_loop.py              # Autonomous LLM agent for research experiments
├── agent_config.py            # Agent configuration and budget settings
├── providers.py               # Multi-LLM provider abstraction
├── train.py                   # Legacy autoresearch training loop (target for agent modifications)
├── finetune/
│   ├── __init__.py
│   ├── trainer.py             # Main QLoRA training runtime (PEFT/TRL + Unsloth)
│   ├── safe_trainer.py        # Safety-hardened SFTTrainer with gradient surgery
│   ├── model_publisher.py     # Release bundle generation and HF publishing
│   ├── zeroth_core.py         # Zeroth-law safety enforcement
│   ├── operability.py         # Model operability checks
│   └── configs/               # Domain-specific YAML configs
│       ├── sps-plc.yaml       # SPS/PLC control systems config
│       ├── legal-dsgvo.yaml   # GDPR/legal domain config
│       └── *.yaml
├── datasets/
│   ├── data_formats.py        # Alpaca/ShareGPT format converters
│   ├── synthetic_generator.py # Synthetic training data generation
│   ├── compiler_feedback.py   # Compiler feedback integration
│   └── legal_data.py          # Legal domain data helpers
├── configs/                   # GPU tier configurations (JSON)
│   ├── tier1-8gb.json
│   ├── tier2-12gb.json
│   └── tier3-24gb.json
├── validation/
│   ├── registry.json          # Hardware validation registry (source of truth)
│   └── *.md                   # Runbooks and validation docs
├── scripts/                   # CI checks and validation scripts
│   ├── check_docs_parity.py   # EN/DE doc synchronization
│   ├── check_repo_hygiene.py  # Repository cleanliness
│   └── validate_*.py          # Various validators
├── tests/                     # pytest test suite
│   ├── test_trainer.py
│   ├── test_agent_loop.py
│   └── test_*.py
├── templates/                 # Model card and governance templates
├── docs/                      # Architecture docs and SOPs
├── docker-compose.finetune.yml
├── Dockerfile.finetune
├── pyproject.toml             # Python package configuration
└── pytest.ini                # Test configuration
```

---

## 4. Build and Development Commands

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -e ".[llm,finetune,dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest -q tests

# Run specific test file
python -m pytest tests/test_trainer.py -v

# Run with coverage
python -m pytest --cov=finetune tests/
```

### Pre-Merge Validation Checks

All checks must pass before merging:

```bash
# Core tests
python -m pytest -q tests

# Documentation checks
python scripts/check_docs_links.py
python scripts/check_docs_parity.py       # EN/DE doc synchronization
python scripts/check_repo_hygiene.py
python scripts/check_template_completeness.py
python scripts/check_compliance_docs.py

# Validation checks
python scripts/validate_validation_registry.py
python scripts/validate_audit_pack.py
python scripts/check_public_claims.py
```

### Docker Operations

```bash
# Fine-tuning pipeline (requires NVIDIA GPU)
AUTORESEARCH_DOMAIN=sps-plc docker compose -f docker-compose.finetune.yml up --build

# Build fine-tune image
docker build -f Dockerfile.finetune -t tuneforge-finetune .
```

### Running Fine-Tuning

```bash
# QLoRA training with YAML config
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --eval

# With backend override
python -m finetune.trainer --config finetune/configs/sps-plc.yaml --backend unsloth --eval

# Agent loop (autonomous research)
python agent_loop.py --provider ollama --model qwen2.5-coder:7b
```

---

## 5. Code Style Guidelines

### Python Style

- **Formatter**: No strict formatter enforced, follow PEP 8
- **Line Length**: 100 characters soft limit
- **Imports**: Standard library → third-party → local modules
- **Type Hints**: Use for public APIs and dataclasses
- **Docstrings**: Google-style docstrings for modules and functions

### Code Patterns

```python
# Dataclass configs with validation
@dataclass
class QLoRAConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    backend: str = "peft_trl"
    
    def __post_init__(self):
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}")

# Provider pattern for LLM abstraction
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], ...) -> str: ...

# Logging with module-level logger
logger = logging.getLogger(__name__)
logger.info("Message: %s", value)
```

### File Organization

- One module per concern (trainer, publisher, data formats)
- CLI entry points at module level with `main()` function
- Tests mirror the source structure

---

## 6. Configuration System

### YAML Training Configs

Located in `finetune/configs/*.yaml`:

```yaml
base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"
output_dir: "output/sps-plc-7b"
backend: "peft_trl"          # or "unsloth"
dataset_path: "datasets/generated/sps"
dataset_format: "alpaca"     # or "sharegpt", "text"
primary_metric: "eval_loss"
metric_goal: "minimize"
lora_r: 32
lora_alpha: 64
learning_rate: 1e-4
max_steps: 2000
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API access |
| `OPENAI_API_KEY` | OpenAI API access |
| `OPENROUTER_API_KEY` | OpenRouter API access |
| `HF_TOKEN` | Hugging Face publishing |
| `AUTORESEARCH_DOMAIN` | Target domain for Docker pipeline |

---

## 7. Testing Strategy

### Test Categories

1. **Unit Tests**: Config parsing, data format conversion, provider abstractions
2. **Integration Tests**: Training pipeline with mocked backends
3. **Validation Tests**: Registry integrity, doc parity, repo hygiene

### Test Conventions

```python
# Tests use pytest fixtures and monkeypatch for mocking
def test_qlora_config_from_yaml(tmp_path):
    yaml_content = "base_model: test\nlora_r: 8\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = QLoRAConfig.from_yaml(config_file)
    assert config.lora_r == 8
```

### GPU Testing

- Most tests mock CUDA operations
- Real GPU tests require `torch.cuda.is_available()`
- Hardware validation runs documented separately in `validation/`

---

## 8. Security Considerations

### Public Repo Rules

- **NO** built Docker images in git
- **NO** real `.env` files or secrets in git
- **NO** generated model artifacts in git (GGUF, safetensors)
- **NO** unverifiable security or compliance claims
- GHCR publication must come from GitHub Actions only

### Secret Handling

```python
# Load from environment, never hardcode
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
```

### Safety Systems

- **Zeroth Core**: Pre-training safety filter (`finetune/zeroth_core.py`)
- **Safe Trainer**: Gradient surgery for safety constraints (`finetune/safe_trainer.py`)
- **Audit Logging**: Structured JSONL audit trails

### Compliance Awareness

- EU AI Act documentation structure (Article 11, 13)
- GDPR/DSGVO data provenance tracking
- Model cards and attestation bundles
- **Important**: Documentation is engineering support, NOT legal certification

---

## 9. Key Architectural Decisions

### Dual Backend Design

The trainer supports both stable `peft_trl` and optimized `unsloth` backends:

```python
if config.backend == "unsloth":
    self._setup_unsloth()
else:
    self._setup_peft_trl()
```

### Provider Abstraction

Only 2 actual implementations cover all LLM providers:
- `AnthropicProvider` — Claude (native SDK)
- `OpenAICompatProvider` — OpenAI, OpenRouter, Kimi, Ollama (OpenAI-compatible)

### Hardware Tier System

Validation registry tracks verification status:
- Tier A: RTX 3090 (24 GB) — Primary public tier
- Tier B: A100/H100 (48 GB+) — Enterprise tier
- Status: `technical_preview` → `verified_rtx_3090` → `verified_48gb_plus`

---

## 10. Common Tasks for Agents

### Adding a New Training Config

1. Create YAML in `finetune/configs/<domain>.yaml`
2. Add test in `tests/test_trainer.py` to verify config loads
3. Update `validation/registry.json` if needed
4. Run pre-merge validation checks

### Adding a New LLM Provider

1. Add provider config to `PROVIDER_REGISTRY` in `providers.py`
2. If OpenAI-compatible: use `OpenAICompatProvider`
3. If unique API: create new provider class inheriting from `LLMProvider`
4. Add tests in `tests/test_providers.py`

### Modifying the Training Loop

1. Core logic is in `finetune/trainer.py` (`QLoRATrainer` class)
2. Safety extensions in `finetune/safe_trainer.py` (`SafeQLoRATrainer`)
3. Update tests for any behavior changes
4. Maintain backward compatibility for YAML configs

### Documentation Updates

- **CRITICAL**: Update both EN and DE versions together
- Required pairs: `README.md`/`README-DE.md`, `CONTRIBUTING.md`/`CONTRIBUTING-DE.md`, etc.
- Run `python scripts/check_docs_parity.py` to verify

---

## 11. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size or use tier1-8gb config |
| Import errors | Ensure `pip install -e ".[finetune]"` |
| Doc parity check fails | Update DE version of modified EN doc |
| GitHub workflow missing | Workflows generated at CI time, not committed |

### Debug Mode

```bash
# Verbose logging
python -m finetune.trainer --config config.yaml --log-level DEBUG

# Dry-run validation
python scripts/validate_validation_registry.py --verbose
```

---

## 12. External Resources

- **Documentation**: `docs/` directory
- **Validation Matrix**: `docs/VALIDATION_MATRIX-EN.md`
- **Compliance**: `COMPLIANCE_STATEMENT.md`
- **Security**: `SECURITY.md`
- **Contributing**: `CONTRIBUTING.md`

---

*This document is a living guide. Update it when architectural decisions change or new patterns emerge.*
