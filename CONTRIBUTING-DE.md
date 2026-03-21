# Mitwirken an TuneForge
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: CONTRIBUTING.md

## Grundsaetze

- TuneForge oeffentlich repo-sicher halten: keine Secrets, keine generierten Artefakte, keine gebauten Images
- Claims an Evidenz binden: kein verifiziertes Label ohne Registry-Beweis
- Doks gespiegelt halten: jedes erforderliche EN-Dokument braucht ein synchrones DE-Pendant
- Compliance sauber begrenzen: auditfaehige Engineering-Unterstuetzung, keine Rechtsberatung

## Pflichtchecks vor dem Merge

- `python -m pytest -q tests`
- `python scripts/check_docs_links.py`
- `python scripts/check_docs_parity.py`
- `python scripts/check_repo_hygiene.py`
- `python scripts/check_template_completeness.py`
- `python scripts/check_compliance_docs.py`
- `python scripts/validate_validation_registry.py`
- `python scripts/check_public_claims.py`
- `python scripts/validate_audit_pack.py`

## Beitragsregeln

- EN und DE fuer erforderliche Public Docs immer gemeinsam aktualisieren
- Templates bei Policy- oder SOP-Aenderungen im gleichen Change nachziehen
- groessere Public-Surface-Aenderungen vor dem Umschreiben archivieren
- fuer neue Repo-Governance-Skripte Tests mitliefern oder aktualisieren
- niemals `.env`, GGUF, Safetensors, Logs oder lokale Outputs committen

Security-Meldungen laufen ueber [SECURITY-DE.md](SECURITY-DE.md). Allgemeines Support-Routing steht in [SUPPORT-DE.md](SUPPORT-DE.md).
