# Fork Attribution

## Upstream Basis

TuneForge is publicly positioned as an open-source self-finetune framework.
Its upstream research basis is [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

- Upstream author: Andrej Karpathy
- Upstream license: MIT
- Preserved upstream source: `upstream/`

## TuneForge-Owned Additions

The following areas are product-owned additions in this repo:

- consumer and workstation GPU packaging
- provider abstraction and multi-provider orchestration
- hybrid QLoRA runtime
- benchmark-first config set for SPS/PLC and Legal/DSGVO
- release bundle generation for Hugging Face and Ollama-compatible packaging
- Docker product surface for TuneForge Studio
- governance, release, compliance, and troubleshooting documentation

## Fork Rules

- Keep upstream code attributable and separable.
- Do not edit `upstream/` for product-only behavior unless the intent is to carry an explicit downstream fork.
- Record all major downstream divergence in `docs/CHANGELOG.md`.
- Keep public docs honest about what is upstream, what is downstream, and what is still benchmark-candidate behavior.
