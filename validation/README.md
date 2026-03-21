# TuneForge Validation Evidence

This directory is the source of truth for public hardware-validation claims.

## Files

- `registry.json`: public validation registry and tier status
- `PRIVATE_PILOT_RUNBOOK.md`: controlled tester instructions
- `templates/environment-manifest.example.json`: environment manifest example
- `templates/tester-attestation.example.json`: tester attestation example
- `templates/failure-report-template.md`: failure-report template for pilot submissions

## Policy

- TuneForge stays in `Technical Preview` until the registry proves both public tiers
- public hardware-verification labels are not allowed without registry evidence
- pilot artifacts must validate before they count toward the registry
