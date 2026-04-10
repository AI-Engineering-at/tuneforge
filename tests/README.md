# TuneForge Test Suite

## Overview

This directory contains the complete test suite for TuneForge.

**Stats:**
- Total tests: 232
- Passing: 232
- Skipped: 2 (1 GPU-dependent, 1 environment-specific)
- Coverage: ~78%

---

## Running Tests

### All Tests

```bash
python -m pytest tests/ -v
```

### Quick Check (quiet mode)

```bash
python -m pytest tests/ -q
```

### With Coverage

```bash
python -m pytest tests/ --cov=finetune --cov=data_utils --cov-report=term-missing
```

### Specific Test File

```bash
python -m pytest tests/test_zeroth_client.py -v
```

### Specific Test

```bash
python -m pytest tests/test_zeroth_client.py::TestZerothClient::test_compute_weights_hash_deterministic -v
```

---

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures
├── test_agent_config.py            # Agent configuration tests
├── test_agent_loop.py              # Main training loop tests
├── test_compiler_feedback.py       # Dataset compiler tests
├── test_data_formats_coverage.py   # Data format tests
├── test_data_loader.py             # Data loading tests
├── test_model_publisher.py         # Model export tests
├── test_operability.py             # State machine/logging tests
├── test_providers.py               # LLM provider tests
├── test_public_repo_governance.py  # Governance checks
├── test_release_tools.py           # Release validation tests
├── test_safe_trainer_math.py       # Gradient surgery math
├── test_safe_trainer_robustness.py # Error handling tests
├── test_synthetic_generator.py     # Synthetic data tests
├── test_trainer.py                 # Main trainer tests
├── test_trainer_coverage.py        # Additional coverage
├── test_validation_tools.py        # Validation registry tests
├── test_zeroth_client.py           # Contract 3 client tests
├── test_zeroth_safety_filter.py    # Safety filter tests
└── integration/                    # Integration tests
    ├── conftest.py
    ├── test_backend_switching.py   # PEFT/TRL ↔ Unsloth
    ├── test_error_handling.py      # Error recovery
    ├── test_export_pipeline.py     # HF → GGUF → Ollama
    └── test_training_pipeline.py   # End-to-end training
```

---

## Test Categories

### Unit Tests

Fast, isolated tests for individual functions/classes.

**Examples:**
- `test_zeroth_client.py` - ZerothClient logic
- `test_operability.py` - StateMachine transitions
- `test_data_formats_coverage.py` - Dataset normalization

### Integration Tests

Tests for component interactions.

**Examples:**
- `test_training_pipeline.py` - Full training flow
- `test_export_pipeline.py` - Export to multiple formats
- `test_backend_switching.py` - Backend compatibility

### Safety Tests

Tests for safety-critical functionality.

**Examples:**
- `test_zeroth_safety_filter.py` - Zeroth integration
- `test_safe_trainer_math.py` - Gradient surgery correctness
- `test_safe_trainer_robustness.py` - Error handling

---

## Skipped Tests

### test_import_hf_datasets_module

**Location:** `tests/test_trainer_coverage.py`

**Reason:** PyArrow environment conflict (not a code bug)

**Skip decorator:**
```python
@pytest.mark.skip(reason="PyArrow environment issue - not a code bug")
```

**Impact:** Low - tests HuggingFace datasets import which works in production

### GPU Tests

**Location:** Various test files

**Pattern:**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
```

**Impact:** None - GPU functionality tested in CI with CUDA runners

---

## Fixtures (conftest.py)

### temp_output_dir

Provides a temporary directory for test outputs.

```python
def test_something(temp_output_dir):
    config = QLoRAConfig(output_dir=str(temp_output_dir))
    # ... test code
```

### sample_dataset_file

Creates a sample dataset file for testing.

```python
def test_loading(sample_dataset_file):
    records = load_jsonl_records(sample_dataset_file)
    # ... test code
```

### sample_config

Provides a minimal QLoRAConfig for testing.

```python
def test_trainer(sample_config):
    trainer = QLoRATrainer(sample_config)
    # ... test code
```

---

## Writing New Tests

### Test Structure

```python
import pytest
from finetune.module import FunctionToTest

class TestFeatureName:
    """Test suite for specific feature."""
    
    def test_happy_path(self):
        """Test normal operation."""
        result = FunctionToTest.call()
        assert result == expected
    
    def test_edge_case(self):
        """Test boundary conditions."""
        result = FunctionToTest.call(boundary_input)
        assert result == expected_boundary
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ExpectedError):
            FunctionToTest.call(invalid_input)
```

### Mocking External Services

```python
from unittest.mock import MagicMock, patch

def test_zeroth_with_mock():
    with patch('finetune.zeroth_client.requests.Session') as mock_session:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "decision": "allow",
            "risk_score": 0.1,
            "reason": "Safe"
        }
        mock_session.return_value.post.return_value = mock_response
        
        client = ZerothClient()
        result = client.evaluate_weight_update(...)
        assert result.allowed
```

---

## Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `data_utils/data_formats.py` | 100% | 100% |
| `finetune/zeroth_client.py` | 95% | 90% |
| `finetune/operability.py` | 90% | 80% |
| `finetune/trainer.py` | 90% | 75% |

---

## CI Integration

Tests run automatically on:
- Pull requests
- Push to main
- Release tags

**Matrix:**
- Python 3.10, 3.11, 3.12
- Ubuntu latest
- Windows latest (partial)

**Required checks:**
- All tests pass
- Coverage ≥ 75%
- Ruff lint clean
- No security warnings (bandit)

---

## Debugging Failed Tests

### Verbose Output

```bash
python -m pytest tests/test_failing.py -v --tb=long
```

### PDB Debugging

```bash
python -m pytest tests/test_failing.py --pdb
```

### Print Debugging

```python
def test_something(capsys):
    print("Debug info")
    captured = capsys.readouterr()
    assert "expected" in captured.out
```

---

## Test Data

### Mock Datasets

Located in `tests/mock_datasets/`:
- `alpaca_small.jsonl` - Sample Alpaca format data
- `sharegpt_small.jsonl` - Sample ShareGPT format data

### Temporary Files

All temporary files created in:
- `tmp/` directory (gitignored)
- Auto-cleaned after test run

---

## Performance Tests

Run separately from unit tests:

```bash
python -m pytest tests/performance/ -v --benchmark-only
```

**Note:** Performance tests require GPU and are skipped in standard runs.

---

## Questions?

See CONTRIBUTING.md for development guidelines.
