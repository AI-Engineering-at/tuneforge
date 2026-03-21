# Archive Manifest — 2026-03-17 Fine-Tune Upgrade

This archive preserves the pre-upgrade state that was superseded by the 2026-03-17
hybrid fine-tune refactor.

## Why this archive exists

- The product moved from a manual fine-tune side path to a loop-compatible hybrid runtime.
- Control configs were retained, but their contract changed.
- User-facing programs and docs were rewritten to reflect the new rollout policy.
- External working notes were updated in place and are mirrored here so the upgrade is auditable.

## Archived repo snapshots

- `README.pre-hybrid-backend.md`
- `finetune-configs/sps-plc.pre-2026-03-17.yaml`
- `finetune-configs/legal-dsgvo.pre-2026-03-17.yaml`
- `programs/program-sps-finetune.pre-2026-03-17.md`
- `programs/program-legal-finetune.pre-2026-03-17.md`

## Archived external working notes

- `notes/framework_v3.pre-2026-03-17.md`
- `notes/v3_review.pre-2026-03-17.md`

## Source of truth

- Repo snapshots were copied from the previous `HEAD` version before the 2026-03-17
  working tree changes.
- External notes were copied from the local pre-upgrade working documents in `Downloads`.
