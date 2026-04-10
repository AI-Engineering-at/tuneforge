# ADR-003: Zeroth-Law Safety Architecture

**Status:** Accepted  
**Date:** 2026-04-10  
**Deciders:** AI Engineering Team

## Context

Fine-tuning LLMs can inadvertently:
- Reduce safety alignments
- Enable harmful outputs
- Cause catastrophic forgetting of safety features

Traditional safety approaches use:
- Post-hoc filters (can be bypassed)
- Prompt engineering (fragile)
- Human review (doesn't scale)

## Decision

Implement **Gradient-Level Safety Enforcement** (Zeroth-Law Architecture):

1. **Pre-Training Safety Check**
   - All training data screened before training
   - Block malicious datasets at source
   - Fail-closed on safety service unavailable

2. **Training-Time Safety**
   - Gradient surgery: orthogonal projection of task gradients against safety gradients
   - Safety constraints don't reduce capability when aligned
   - Mathematical guarantee of safety preservation

3. **Pre-Publish Safety Check**
   - Model card validation
   - Safety evaluation benchmarks
   - Block publish if safety degraded

## Mathematical Foundation

```
Given:
- g_task: gradient from task loss
- g_safety: gradient from safety constraint

If g_task · g_safety < 0 (conflict):
    g_final = g_task - proj(g_task onto g_safety)
Else:
    g_final = g_task (synergy, no intervention)
```

## Consequences

### Positive

- Provable safety preservation
- No inference overhead
- Automatic enforcement
- Scales with model size

### Negative

- Requires safety reference gradients
- Computational cost during training
- Complex implementation

## Implementation

- `finetune/zeroth_core.py`: Safety API integration
- `finetune/safe_trainer.py`: Gradient surgery
- `finetune/operability.py`: State machine for safety events

## References

- [ZEROTH_HARDENING_ARCHITECTURE.md](../../ZEROTH_HARDENING_ARCHITECTURE.md)
- `finetune/safe_trainer.py`
