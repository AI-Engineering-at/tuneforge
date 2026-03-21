# Security Policy

## Scope

This policy covers the TuneForge product surface in `products/tuneforge`, including:

- source code
- Docker images
- model publishing helpers
- release manifests
- product documentation

It does not cover private infrastructure, internal open-notebook content, or third-party hosted services outside our control.

## Supported Release Surface

- current `main`
- the latest tagged TuneForge release
- the latest tagged TuneForge Studio Docker image

## Reporting

Report security issues privately before public disclosure.

- Preferred channel: direct responsible disclosure to the maintainers of this repo
- Include: reproduction steps, affected files or workflow, expected impact, and whether secrets or model artifacts are exposed

Do not open a public issue for a live secret, release-signing, or supply-chain problem.

## Release Hardening Expectations

TuneForge public releases should include:

- CI pass on tests and compile checks
- release artifact validation
- third-party attribution verification
- SBOM generation for Docker releases
- signed container publication for GHCR releases

## Non-Goals

- no warranty of regulatory compliance by default
- no guarantee that published models are safe for high-risk use
- no support for undisclosed customer-specific data handling claims
