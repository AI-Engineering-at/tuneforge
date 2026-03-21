# TuneForge Model Release Policy

Current default release state: **Technical Preview**.

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

Preview model releases may be published only if they remain explicitly labeled `Technical Preview` and do not claim verified hardware status.

## Required Bundle

- `README.md`
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `environment-manifest.json`
- `tester-attestation.json`
- `validation-result.json`
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
- verified hardware labels without matching entries in `validation/registry.json`

## Review Requirements

- one engineering review for artifact correctness
- one governance review for model card, provenance, and licensing
- explicit keep-or-publish decision recorded in the release notes
