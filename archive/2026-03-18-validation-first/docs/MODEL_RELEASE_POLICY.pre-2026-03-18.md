# TuneForge Model Release Policy

## Public Model Release Gate

No TuneForge model release is public until all of the following exist:

- validated benchmark result
- base model license recorded
- dataset provenance recorded
- intended use and out-of-scope use documented
- limitations and safety notes documented
- hardware and runtime budget documented
- benchmark summary attached
- release bundle generated from the exact artifact being published

## Required Bundle

- `README.md`
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `Modelfile` for Ollama-compatible packaging when GGUF is produced

## Allowed Public Targets

- GitHub Release assets
- Hugging Face adapter repo
- GGUF plus `Modelfile` for Ollama-compatible distribution

## Disallowed Public Claims

- unverified accuracy or compliance claims
- "enterprise ready" without governance artifacts
- "DSGVO compliant" as a legal guarantee
- benchmark comparisons across unmatched hardware budgets

## Review Requirements

- one engineering review for artifact correctness
- one governance review for model card, provenance, and licensing
- explicit keep-or-publish decision recorded in the release notes
