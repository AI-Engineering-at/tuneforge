# Contributing Guide

Thank you for your interest in contributing to TuneForge!

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 12.4+ (for GPU testing)
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/AI-Engineerings-at/tuneforge.git
cd tuneforge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,finetune,llm]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests
python -m pytest tests/ -q

# Check code style
ruff check .
mypy finetune/
```

## Development Workflow

### Branch Strategy

```
main          - Production-ready code
  ↓
develop       - Integration branch
  ↓
feature/*     - New features
bugfix/*      - Bug fixes
docs/*        - Documentation
```

### Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes:**
   - Follow code style guidelines
   - Add/update tests
   - Update documentation

3. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   pytest --cov=finetune tests/  # With coverage
   ```

4. **Check documentation parity (EN/DE):**
   ```bash
   python scripts/check_docs_parity.py
   ```

5. **Commit:**
   ```bash
   git commit -m "feat: add new feature"
   ```

6. **Push and create PR:**
   ```bash
   git push origin feature/my-feature
   # Create PR on GitHub
   ```

## Code Style

### Python Style

- **Formatter:** Ruff
- **Line Length:** 100 characters
- **Type Hints:** Required for public APIs

```python
# Good
def train_model(config: QLoRAConfig) -> TrainingSummary:
    """Train a model with given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Training summary with metrics
        
    Raises:
        ValueError: If config is invalid
    """
    if not config.base_model:
        raise ValueError("base_model required")
    return trainer.train()

# Bad
def train_model(config):
    """Train model."""
    return trainer.train(config)
```

### Import Order

```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party
import torch
import yaml
from transformers import AutoModel

# 3. Local
from finetune.trainer import QLoRATrainer
from data_utils.data_formats import normalize_records
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `QLoRATrainer` |
| Functions | snake_case | `train_model()` |
| Constants | UPPER_SNAKE | `MAX_STEPS` |
| Private | _leading_underscore | `_internal_helper()` |

## Testing

### Test Structure

```
tests/
├── test_module_name.py      # Unit tests
├── test_integration/        # Integration tests
│   ├── test_pipeline.py
│   └── conftest.py
└── conftest.py              # Shared fixtures
```

### Writing Tests

```python
# tests/test_trainer.py
import pytest
from finetune.trainer import QLoRAConfig


def test_config_defaults():
    """Test default configuration values."""
    config = QLoRAConfig()
    assert config.backend == "peft_trl"
    assert config.lora_r == 16


def test_invalid_backend_raises():
    """Test that invalid backend raises ValueError."""
    config = QLoRAConfig()
    config.backend = "invalid"
    
    with pytest.raises(ValueError, match="Unsupported backend"):
        validate_config(config)
```

### Test Coverage

- Minimum 80% coverage for new code
- 100% coverage for critical paths (safety, validation)
- Integration tests for end-to-end workflows

## Documentation

### Documentation Types

| Type | Location | Audience |
|------|----------|----------|
| Code docs | Docstrings | Developers |
| API docs | docs/api/ | Users |
| Guides | docs/guide/ | Users |
| ADRs | docs/architecture/adr/ | Architects |

### Docstring Format (Google Style)

```python
def function(arg1: str, arg2: int) -> bool:
    """Short description.
    
    Longer description if needed. Can span
    multiple lines.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg2 is negative
        
    Example:
        >>> function("test", 42)
        True
    """
    if arg2 < 0:
        raise ValueError("arg2 must be non-negative")
    return True
```

### Documentation Parity (EN/DE)

All user-facing documentation must be bilingual:

```
README.md           → README-DE.md
CONTRIBUTING.md     → CONTRIBUTING-DE.md
docs/guide/*.md     → docs/guide/*-DE.md
```

Check with:
```bash
python scripts/check_docs_parity.py
```

## Pull Request Guidelines

### PR Title Format

```
feat: add new feature
fix: correct bug in trainer
docs: update installation guide
test: add integration tests
refactor: simplify config handling
security: fix vulnerability
```

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Coverage maintained (`pytest --cov`)
- [ ] Code style passes (`ruff check .`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated
- [ ] EN/DE parity maintained
- [ ] CHANGELOG.md updated

### Review Process

1. **Automated checks** must pass (CI)
2. **Code review** by at least 1 maintainer
3. **Documentation review** if docs changed
4. **Security review** if security-related

## Release Process

### Version Numbering

Semantic Versioning: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features, backward compatible
- **PATCH:** Bug fixes

### Creating a Release

1. **Update version:**
   ```bash
   # Update pyproject.toml
   version = "1.1.0"
   ```

2. **Update CHANGELOG:**
   ```markdown
   ## 1.1.0 - 2026-04-10
   
   ### Added
   - New feature X
   
   ### Fixed
   - Bug Y
   ```

3. **Create tag:**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

4. **CI builds release** automatically

## Getting Help

- **Discussions:** GitHub Discussions
- **Issues:** GitHub Issues (bugs, features)
- **Security:** security@ai-engineering.at

## Code of Conduct

See [CODE_OF_CONDUCT.md](../../CODE_OF_CONDUCT.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
