# TuneForge Release Policy

## Purpose

This policy defines what counts as a public TuneForge release and what evidence must exist before publication.

## Release Types

### Code release

- tagged Git commit
- changelog entry
- passing CI
- archived pre-release snapshot for major structural changes

### Docker release

- GHCR image target: `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- build provenance and SBOM
- checksums for attached artifacts
- signed image publication

### Model release

- benchmark evidence on declared hardware
- `README.md` / Model Card
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `license-manifest`
- explicit release approval

## Benchmark-First Rule

- controls remain default until a candidate wins on the same metric and hardware budget
- benchmark claims must include hardware, dataset, metric, and runtime context
- unpublished or failed runs cannot be used in public claims

## Change Management

- breaking public changes require a changelog entry and migration note
- branding, licensing, or release process changes require an archive snapshot
- release automation must fail closed on missing manifests or missing attribution
