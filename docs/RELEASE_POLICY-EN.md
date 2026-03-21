# TuneForge Release Policy
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: RELEASE_POLICY-DE.md

Current product state: **Technical Preview**.

## Release Types

### Code release

- tagged Git commit
- changelog entry
- passing CI
- public-repo readiness checks
- allowed while the product remains in preview

### Docker preview release

- canonical image target: `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- built in GitHub Actions
- GHCR publication only from workflow automation
- SBOM, checksums, and signed metadata required

### Model release

- complete release bundle
- validation registry compatible status
- model card and provenance docs complete
- explicit approval decision recorded
- no verified hardware label without registry proof

## Required Gates

- code quality gate
- docs parity gate
- repo hygiene gate
- compliance docs gate
- template completeness gate
- audit pack gate
- validation registry gate
- public claim gate

## Benchmark-First Rule

- controls remain default until a candidate wins on the same hardware budget and metric
- published comparisons must include hardware, dataset, runtime context, and public status
- unpublished or failed runs may inform internal learning but may not be used for public claims

## Public Repo Boundaries

- no built images in git
- no secrets in git
- no generated model artifacts in git
- no compliance guarantee language
- no verified hardware labels without evidence

## Change Management

- structural public-surface changes require an archive snapshot
- breaking release-process changes require migration notes
- emergency fixes follow [PATCH_PLAN-EN.md](PATCH_PLAN-EN.md)
- strategic platform changes follow [UPGRADE_PLAN-EN.md](UPGRADE_PLAN-EN.md)
