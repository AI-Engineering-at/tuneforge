# TuneForge Upgrade Plan
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: UPGRADE_PLAN-DE.md

## Purpose

Define how TuneForge evolves through controlled platform upgrades without losing reproducibility, public trust, or registry compatibility.

## Upgrade Classes

### Platform upgrade

- example: new public repo layout, new Docker release discipline, new validation model
- requires archive snapshot
- requires changelog entry
- requires migration note

### Training stack upgrade

- example: new base-model family, trainer backend change, dependency floor increase
- requires benchmark comparison against the current control
- requires updated templates or SOPs when release evidence changes

### Governance upgrade

- example: new compliance template, new audit schema, new release evidence rule
- requires DE and EN doc updates together
- requires CI readiness checks to stay green

## Upgrade Sequence

1. Archive the pre-upgrade public surface.
2. Document scope, risk, rollback, and acceptance criteria.
3. Update code and documentation in the same change set.
4. Run full CI and public-repo readiness checks.
5. Revalidate release-bundle outputs if manifests changed.
6. Publish only after changelog and migration notes are complete.

## Upgrade Guardrails

- do not change the default control without matched benchmark evidence
- do not change public claim language without validation and compliance review
- do not remove required release artifacts without updating validators and policies
