# Changelog

## 2026-03-18

### Validation-first release discipline

- Marked TuneForge public surfaces as `Technical Preview` until hardware proof exists
- Added a validation matrix and private-pilot proof workflow for RTX 3090 and 48 GB+ tiers
- Expanded release bundles with environment manifests, tester attestations, and validation results
- Added registry- and claim-check tooling so public docs cannot claim more than the evidence proves

## 2026-03-17

### TuneForge public surface

- Repositioned the public product surface from `autoresearch-kit` to `TuneForge`
- Defined `TuneForge Studio` as the Docker product surface
- Added product-level MIT licensing for this product directory
- Added public governance and attribution documents:
  - `THIRD_PARTY.md`
  - `SECURITY.md`
  - `docs/RELEASE_POLICY.md`
  - `docs/MODEL_RELEASE_POLICY.md`
  - `docs/COMPLIANCE_PACK.md`

### Fine-tune runtime

- Added hybrid backend support in `finetune/trainer.py`
- Added supported CLI entrypoint: `python -m finetune.trainer`
- Added dataset normalization to a single `text` training field
- Added machine-readable metric output for release bundles and loop integration

### Autoresearch loop

- Generalized keep/discard logic to support configurable `primary_metric` and `metric_goal`
- Kept `val_bpb` backward-compatible for the original autoresearch path

### Release discipline

- Added release-bundle generation and publishing groundwork for GitHub, Hugging Face, and Ollama-compatible outputs
- Added archive snapshot for the pre-TuneForge public surface under `archive/2026-03-17-finetune-upgrade/`
