# TuneForge Architecture
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: ARCHITECTURE-DE.md

Current public status: **Technical Preview**.

## Architectural Position

TuneForge is a layered upgrade on top of `karpathy/autoresearch`, not a rewrite.

- `autoresearch` remains the research-loop foundation
- TuneForge adds local fine-tuning, governed publication, validation evidence, and public-repo discipline
- the main public target is single-GPU RTX 3090 first, with 48 GB+ as the secondary verified tier

## Runtime Layers

### Research loop

- source of truth: `agent_loop.py`, `agent_config.py`, `providers.py`
- purpose: propose, run, compare, and keep or discard experiments
- legacy metric compatibility: `val_bpb`

### Fine-tune runtime

- source of truth: `finetune/trainer.py`
- backends: `peft_trl` as the stable path, `unsloth` as the optional benchmark path
- public config surface: backend, dataset format, metric direction, rsLoRA flag, release metrics

### Release and publishing runtime

- source of truth: `finetune/model_publisher.py`
- outputs: model card, training manifest, benchmark summary, environment manifest, tester attestation, validation result, license manifest, optional `Modelfile`
- public publication targets: GitHub Releases, Hugging Face, Ollama-compatible GGUF packaging

### Validation and proof layer

- source of truth: `validation/registry.json`
- purpose: gate public hardware claims and model-promotion claims
- proof rule: two independent successful end-to-end runs per public hardware tier

### Audit and governance layer

- source of truth: [LOGGING_AUDIT_PROTOCOL-EN.md](LOGGING_AUDIT_PROTOCOL-EN.md), [COMPLIANCE_PACK-EN.md](COMPLIANCE_PACK-EN.md), [RELEASE_POLICY-EN.md](RELEASE_POLICY-EN.md)
- purpose: keep the repo audit-ready, attribution-safe, and evidence-bound

## Public Interfaces

- CLI: `python -m finetune.trainer`
- release bundle manifests: `training-manifest.json`, `benchmark-summary.json`, `environment-manifest.json`, `tester-attestation.json`, `validation-result.json`
- machine log: `results/protocol.jsonl`
- registry: `validation/registry.json`

## Knowledge Boundaries

- private source of truth: open-notebook
- public exports: docs, templates, benchmark evidence, release bundles, blog-ready content
- referenced subsystem for legal-source provenance: [LEGAL_SOURCE_REFERENCES-EN.md](LEGAL_SOURCE_REFERENCES-EN.md)

## Operating Rules

- shipped controls remain default until a candidate wins on the same metric and hardware budget
- 3090 is the public reference tier for v1
- stronger hardware is supported, but claims must stay hardware-specific
- TuneForge stays in preview until public evidence proves the validation matrix
