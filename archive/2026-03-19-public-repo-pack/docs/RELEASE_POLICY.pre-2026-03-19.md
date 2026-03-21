# TuneForge Release Policy

## Purpose

This policy defines what counts as a public TuneForge release and what evidence must exist before publication.

Current product status: **Technical Preview** until the validation registry proves both public hardware tiers.

## Release Types

### Code release

- tagged Git commit
- changelog entry
- passing CI
- archived pre-release snapshot for major structural changes
- may ship as a Technical Preview after CI is green

### Docker release

- GHCR image target: `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- build provenance and SBOM
- checksums for attached artifacts
- signed image publication
- may ship as a Technical Preview after CI is green

### Model release

- benchmark evidence on declared hardware
- `README.md` / Model Card
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `environment-manifest.json`
- `tester-attestation.json`
- `validation-result.json`
- `license-manifest.json`
- explicit release approval
- verified public hardware labels require registry proof, not a single successful run

## Validation Gates

- code quality gate: tests, compile checks, attribution, docs, and public-claim checks
- release bundle gate: required bundle artifacts plus `validation-result.json`
- hardware validation gate: two independent successful runs per public tier
- publishing gate: no GitHub, Hugging Face, Docker, or Ollama claim may exceed the registry evidence

## Benchmark-First Rule

- controls remain default until a candidate wins on the same metric and hardware budget
- benchmark claims must include hardware, dataset, metric, and runtime context
- unpublished or failed runs cannot be used in public claims
- unverified pilot runs cannot be used for public hardware-verification labels

## Change Management

- breaking public changes require a changelog entry and migration note
- branding, licensing, or release process changes require an archive snapshot
- release automation must fail closed on missing manifests or missing attribution
