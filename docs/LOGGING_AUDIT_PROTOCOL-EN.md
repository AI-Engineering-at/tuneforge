# TuneForge Logging and Audit Protocol
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: LOGGING_AUDIT_PROTOCOL-DE.md

## Purpose

Formalize TuneForge runtime and release traceability into a public audit-ready contract.

## Canonical Machine Log

- file: `results/protocol.jsonl`
- format: one JSON object per line
- schema reference: `audit/protocol.schema.json`

## Minimum Protocol Fields

- `timestamp`
- `run_id`
- `event_type`
- `stage`
- `status`
- `message`
- `git_sha`

Optional but recommended:

- `hardware_tier`
- `public_status`
- `metric_name`
- `metric_value`
- `artifact_path`
- `error_id`

## Event Taxonomy

- `run.started`, `run.finished`, `run.failed`
- `validation.started`, `validation.accepted`, `validation.rejected`
- `publish.started`, `publish.completed`, `publish.failed`
- `incident.opened`, `incident.updated`, `incident.closed`

## Audit Records

- error registry: recurring or known failure classes
- incident log: significant operational failures and recovery actions
- release decision log: keep, publish, delay, rollback decisions
- SIEM export schema: `audit/siem-export.schema.json`

## Redaction and Retention

- do not log secrets
- do not log private customer content into public artifacts
- keep enough event context for audit and rollback
- retain public release evidence as long as the release remains public

## Mapping Rules

- every public validation decision should map to a validation event plus a registry update
- every public model release should map to publish events plus a release decision entry
- every recurring failure should map to an error-registry entry and a troubleshooting note
