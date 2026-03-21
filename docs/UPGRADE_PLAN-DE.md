# TuneForge Upgrade-Plan
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: UPGRADE_PLAN-EN.md

## Zweck

Festlegen, wie sich TuneForge ueber kontrollierte Plattform-Upgrades weiterentwickelt, ohne Reproduzierbarkeit, oeffentliches Vertrauen oder Registry-Kompatibilitaet zu verlieren.

## Upgrade-Klassen

### Plattform-Upgrade

- Beispiel: neues Public-Repo-Layout, neue Docker-Release-Disziplin, neues Validation-Modell
- braucht Archive-Snapshot
- braucht Changelog-Eintrag
- braucht Migrationshinweis

### Trainings-Stack-Upgrade

- Beispiel: neue Basismodell-Familie, Backend-Wechsel im Trainer, hoehere Dependency-Untergrenze
- braucht Benchmark-Vergleich gegen die aktuelle Control
- braucht aktualisierte Templates oder SOPs, wenn sich Release-Evidence aendert

### Governance-Upgrade

- Beispiel: neues Compliance-Template, neues Audit-Schema, neue Release-Evidence-Regel
- braucht gemeinsame DE- und EN-Doku-Updates
- braucht gruene CI- und Readiness-Checks

## Upgrade-Ablauf

1. Die Public Surface vor dem Upgrade archivieren.
2. Umfang, Risiko, Rollback und Acceptance Criteria dokumentieren.
3. Code und Dokumentation im selben Change Set aktualisieren.
4. Vollstaendige CI- und Public-Repo-Readiness-Checks ausfuehren.
5. Release-Bundle-Outputs neu validieren, wenn sich Manifeste geaendert haben.
6. Erst nach vollstaendigem Changelog und Migrationshinweisen publizieren.

## Upgrade-Gelander

- den Default-Control nicht ohne gematchte Benchmark-Evidence aendern
- oeffentliche Claim-Sprache nicht ohne Validation- und Compliance-Review aendern
- erforderliche Release-Artefakte nicht entfernen, ohne Validatoren und Policies mit anzupassen
