# TuneForge Security Policy
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: SECURITY-DE.md

TuneForge is a public engineering framework and Docker-based product surface. Security handling must stay compatible with a future dedicated public GitHub repo and with GHCR-based release publication.

## Scope

This policy covers the public TuneForge surface in `products/tuneforge`, including:

- source code
- Dockerfiles and compose files
- GitHub Actions workflows
- release bundle tooling
- public documentation and templates
- validation registry and audit artifacts

It does not cover private infrastructure, private open-notebook content, customer environments, or third-party services outside our control.

## Supported Surface

- current `main`
- latest tagged TuneForge code release
- latest tagged TuneForge Studio Docker preview release on GHCR

Current public status remains **Technical Preview**.

## Reporting

Report security issues privately before public disclosure.

- include affected file, workflow, or command path
- include reproduction steps and expected impact
- state whether secrets, provenance data, or release integrity are involved
- do not open a public issue for live credentials, signing, or supply-chain findings

## Public Repo Rules

- no built Docker images in git
- no real `.env` files or secrets in git
- no generated model artifacts in git
- no unverifiable security or compliance claims
- GHCR publication must come from GitHub Actions, not from committed artifacts

## Release Hardening Expectations

Public TuneForge releases should include:

- passing CI and public-repo readiness checks
- release artifact validation
- attribution and license validation
- SBOM generation for GHCR Docker releases
- checksum generation and signed metadata for GHCR release outputs

## Boundaries

- TuneForge documentation is governance and engineering support
- it is not legal advice
- it is not a security certification
- it does not guarantee Austria, DSGVO, or EU AI Act compliance by default
