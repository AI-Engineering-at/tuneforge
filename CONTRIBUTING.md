# Contributing to TuneForge
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: CONTRIBUTING-DE.md

## Principles

- keep TuneForge public-repo safe: no secrets, no generated artifacts, no built images
- keep claims evidence-bound: no verified label without registry proof
- keep docs mirrored: every required EN document must have a synchronized DE twin
- keep compliance bounded: audit-ready engineering support, not legal advice

## Required Checks Before Merge

- `python -m pytest -q tests`
- `python scripts/check_docs_links.py`
- `python scripts/check_docs_parity.py`
- `python scripts/check_repo_hygiene.py`
- `python scripts/check_template_completeness.py`
- `python scripts/check_compliance_docs.py`
- `python scripts/validate_validation_registry.py`
- `python scripts/check_public_claims.py`
- `python scripts/validate_audit_pack.py`

## Contribution Rules

- update EN and DE together for required public docs
- update templates in the same change as policy or SOP changes
- archive major public-surface changes before rewriting them
- add or update tests for any new repo-governance script
- do not commit `.env`, GGUF, safetensors, logs, or local outputs

Security disclosures follow [SECURITY.md](SECURITY.md). General support routing lives in [SUPPORT.md](SUPPORT.md).
