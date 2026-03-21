# TuneForge Logging- und Audit-Protokoll
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: LOGGING_AUDIT_PROTOCOL-EN.md

## Zweck

Die Runtime- und Release-Traceability von TuneForge in einen oeffentlichen audit-ready Vertrag ueberfuehren.

## Kanonisches Maschinenlog

- Datei: `results/protocol.jsonl`
- Format: ein JSON-Objekt pro Zeile
- Schema-Referenz: `audit/protocol.schema.json`

## Minimale Protocol-Felder

- `timestamp`
- `run_id`
- `event_type`
- `stage`
- `status`
- `message`
- `git_sha`

Optional, aber empfohlen:

- `hardware_tier`
- `public_status`
- `metric_name`
- `metric_value`
- `artifact_path`
- `error_id`

## Event-Taxonomie

- `run.started`, `run.finished`, `run.failed`
- `validation.started`, `validation.accepted`, `validation.rejected`
- `publish.started`, `publish.completed`, `publish.failed`
- `incident.opened`, `incident.updated`, `incident.closed`

## Audit-Records

- Error Registry: wiederkehrende oder bekannte Fehlerklassen
- Incident Log: bedeutende betriebliche Fehler und Recovery-Massnahmen
- Release Decision Log: `keep`, `publish`, `delay`, `rollback`
- SIEM-Export-Schema: `audit/siem-export.schema.json`

## Redaction und Retention

- keine Secrets loggen
- keine privaten Kundeninhalte in oeffentliche Artefakte loggen
- genug Event-Kontext fuer Audit und Rollback behalten
- oeffentliche Release-Evidence so lange halten, wie der Release oeffentlich verfuegbar ist

## Mapping-Regeln

- jede oeffentliche Validation-Entscheidung sollte auf ein Validation-Event plus Registry-Update abgebildet werden
- jeder oeffentliche Modell-Release sollte auf Publish-Events plus einen Release-Decision-Eintrag abgebildet werden
- jeder wiederkehrende Fehler sollte auf einen Error-Registry-Eintrag und einen Troubleshooting-Hinweis abgebildet werden
