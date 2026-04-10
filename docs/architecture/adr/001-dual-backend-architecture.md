# ADR-001: Dual Backend Architecture (PEFT/TRL + Unsloth)

**Status:** Accepted  
**Date:** 2026-04-10  
**Deciders:** AI Engineering Team

## Context

TuneForge needs to support both stable, well-tested training backends and cutting-edge optimized backends. Users have different requirements:

- **Enterprise users** need stability, reproducibility, and extensive documentation
- **Research users** want maximum speed and latest optimizations
- **Validation** requires a stable reference implementation

## Decision

Implement a **Dual Backend Architecture** with:

1. **Primary Backend:** PEFT + TRL (HuggingFace official)
   - Stable, well-documented
   - Reference for validation
   - Default choice

2. **Secondary Backend:** Unsloth (optimized)
   - 2-5x faster training
   - 70% less memory
   - Optional, user-selected

Both backends share the same:
- Configuration interface
- Dataset processing
- Export pipeline
- Evaluation metrics

## Consequences

### Positive

- Users can choose speed vs. stability
- Validation uses stable backend
- Easy to add future backends
- Consistent user experience

### Negative

- More code to maintain
- Testing matrix doubles
- Backend-specific bugs possible

## Implementation

```python
class QLoRATrainer:
    def __init__(self, config: QLoRAConfig):
        if config.backend == "unsloth":
            self.backend = UnslothBackend()
        else:
            self.backend = PEFTRLBackend()
```

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [TRL Documentation](https://huggingface.co/docs/trl/)
