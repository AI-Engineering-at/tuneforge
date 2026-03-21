# TuneForge Validation Matrix
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: VALIDATION_MATRIX-DE.md

Current public status: **Technical Preview**.

## Public Hardware Tiers

| Tier | Label | Scope | Verification Rule |
|------|-------|-------|-------------------|
| Tier A | RTX 3090 / 24 GB | primary public fine-tune and benchmark tier | 2 independent successful end-to-end runs |
| Tier B | 48 GB+ | secondary enterprise-scale validation tier | 2 independent successful end-to-end runs |

## End-to-End Proof Requirement

Every qualifying run must prove:

- Docker start
- model download
- fine-tune launch
- evaluation
- machine-readable metrics
- release-bundle generation
- release-bundle validation
- Hugging Face publish dry run or approved real publish
- Ollama package dry run when GGUF packaging is part of the run

## Model Scope by Tier

- Tier A validates the shipped controls and `Qwen/Qwen3-8B`
- Tier B validates the same 8B path plus one 14B-class candidate
- `Mistral Small 4` stays documentation-only for v1

## Claim Rules

- public docs may always say `Technical Preview`
- no public doc may use a verified hardware label before the registry is verified
- unpublished, failed, or unmatched-budget runs do not count as public benchmark evidence
- candidate promotion must happen on the same metric and same hardware budget as the control

## Evidence Registry

The source of truth is [../validation/registry.json](../validation/registry.json).

Each accepted run must record:

- hardware tier
- model and config
- run date
- tester identifier
- result status
- independent environment flag
- artifact path or evidence reference

## Private Pilot

External hardware proof is collected through the private pilot:

- [../validation/PRIVATE_PILOT_RUNBOOK-EN.md](../validation/PRIVATE_PILOT_RUNBOOK-EN.md)
- `validation/templates/environment-manifest.example.json`
- `validation/templates/tester-attestation.example.json`
- `validation/templates/failure-report-template.md`
