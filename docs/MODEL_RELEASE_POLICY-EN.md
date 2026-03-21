# TuneForge Model Release Policy
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: MODEL_RELEASE_POLICY-DE.md

Current default release state: **Technical Preview**.

## Required Bundle

Every public model release must provide:

- `README.md`
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `license-manifest.json`
- `environment-manifest.json`
- `tester-attestation.json`
- `validation-result.json`
- `Modelfile` when GGUF packaging is produced

## Required Documentation

- completed model card
- dataset provenance record
- benchmark evidence record
- release approval decision
- risk classification and intended-use assessment
- DPIA or VVT input when the deployment context requires it

## Allowed Targets

- GitHub release assets
- Hugging Face adapter repositories
- Ollama-compatible GGUF packaging with `Modelfile`

## Disallowed Claims

- no legal or compliance certification language
- no verified hardware label without matching registry proof
- no benchmark win claim without matched hardware budget
- no vague model-quality claim without evidence attached

## Review Flow

- engineering review for artifact integrity
- governance review for provenance, licensing, and documentation completeness
- explicit keep, delay, publish, or rollback decision in the release decision log

## Public Naming

- Hugging Face adapter convention: `<org>/tuneforge-<domain>-<base>-adapter`
- Ollama convention: `tuneforge:<domain>-<base>-<quant>`
