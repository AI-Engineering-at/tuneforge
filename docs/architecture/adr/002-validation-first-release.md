# ADR-002: Validation-First Release Discipline

**Status:** Accepted  
**Date:** 2026-04-10  
**Deciders:** AI Engineering Team

## Context

Many ML frameworks make claims about performance without reproducible evidence. This leads to:

- Unreliable benchmark comparisons
- User frustration with "works on my machine"
- Regulatory compliance issues

## Decision

Implement a **Validation-First Release Discipline**:

1. **No hardware claims without validation**
   - Every supported GPU tier needs 2+ successful runs
   - Results must be within ±2% tolerance
   - Evidence must be public

2. **Validation Registry**
   - Central `validation/registry.json`
   - Records every validation run
   - Tiers move from "unverified" to "verified"

3. **Release Gating**
   - Major releases require Tier A verification
   - Cannot claim "production ready" without validation

## Consequences

### Positive

- Trust through transparency
- Reproducible results
- Clear hardware requirements
- Compliance-ready evidence

### Negative

- Slower release cycles
- Hardware dependency for releases
- More documentation overhead

## Registry Format

```json
{
  "tiers": {
    "tier_a_rtx_3090_24gb": {
      "status": "verified",
      "required_successful_runs": 2
    }
  },
  "runs": [
    {
      "tier": "tier_a_rtx_3090_24gb",
      "date": "2026-04-10",
      "result": "success",
      "evidence_path": "validation/evidence/..."
    }
  ]
}
```

## References

- [VALIDATION_MATRIX-EN.md](../../VALIDATION_MATRIX-EN.md)
- [TIER_A_VALIDATION_PLAN.md](../../../validation/TIER_A_VALIDATION_PLAN.md)
