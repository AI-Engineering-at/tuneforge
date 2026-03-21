# TuneForge Patch Plan
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: PATCH_PLAN-DE.md

## Purpose

Define how TuneForge handles urgent patches without turning the public repo into an unreviewed hotfix stream.

## Patch Classes

### P0 security or integrity patch

- live secret exposure
- release-signing failure
- broken artifact validator
- broken public-claim gate

### P1 runtime or publishing patch

- trainer CLI regression
- release-bundle regression
- broken model-publish workflow

### P2 documentation or governance patch

- missing DE or EN mirror
- outdated compliance token
- broken template link

## Patch Procedure

1. Open a tracked issue or internal incident entry.
2. Record scope, affected files, and rollback path.
3. Apply the smallest viable fix.
4. Run the full relevant validation set.
5. Update changelog and release decision log.
6. Archive the public surface if the patch changes release policy or public contracts.

## Patch Constraints

- no silent policy changes
- no unreviewed release automation changes
- no patch may add new public claims without evidence
- hotfixes still need EN and DE parity for required public docs
