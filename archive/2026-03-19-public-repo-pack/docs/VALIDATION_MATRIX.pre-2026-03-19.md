# TuneForge Validation Matrix

## Current Status

TuneForge is currently a **Technical Preview**.

The product may ship code, Docker previews, and preview model bundles, but it may not claim verified hardware support until the validation registry records enough independent successful runs.

## Public Hardware Tiers

| Tier | Label | Requirement | Current Status |
|------|-------|-------------|----------------|
| Tier A | RTX 3090 / 24 GB | 2 independent successful end-to-end runs | Unverified |
| Tier B | 48 GB+ | 2 independent successful end-to-end runs | Unverified |

## End-to-End Proof Requirement

Every qualifying run must prove:

- Docker start
- model download
- fine-tune launch
- evaluation
- machine-readable metrics
- release bundle generation
- Hugging Face publish dry run or approved real publish
- Ollama package dry run

## Model Scope by Tier

- Tier A validates shipped controls and `Qwen/Qwen3-8B`
- Tier B validates the same 8B path plus one official 14B candidate
- `Mistral Small 4` remains documentation-only for v1 and is not part of the runnable validation matrix

## Registry and Claims

- Source of truth: `validation/registry.json`
- public docs may use `Technical Preview`
- hardware-verification labels are gated by the registry
- unmatched hardware comparisons and unpublished runs do not count

## Private Pilot

Validation evidence is expected to come from a controlled private pilot until in-house hardware proof exists.

Use:

- `validation/PRIVATE_PILOT_RUNBOOK.md`
- `validation/templates/environment-manifest.example.json`
- `validation/templates/tester-attestation.example.json`
