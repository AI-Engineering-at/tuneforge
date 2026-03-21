# TuneForge Patch-Plan
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: PATCH_PLAN-EN.md

## Zweck

Festlegen, wie TuneForge dringende Patches behandelt, ohne das Public Repo in einen unreviewten Hotfix-Stream zu verwandeln.

## Patch-Klassen

### P0 Security- oder Integritaets-Patch

- Live-Secret-Exposure
- Release-Signing-Fehler
- defekter Artefakt-Validator
- defektes Public-Claim-Gate

### P1 Runtime- oder Publishing-Patch

- Regression der Trainer-CLI
- Regression im Release-Bundle
- defekter Model-Publish-Workflow

### P2 Dokumentations- oder Governance-Patch

- fehlender DE- oder EN-Spiegel
- veraltetes Compliance-Token
- kaputter Template-Link

## Patch-Ablauf

1. Eine nachverfolgbare Issue oder einen internen Incident-Eintrag anlegen.
2. Umfang, betroffene Dateien und Rollback-Pfad erfassen.
3. Den kleinstmoeglichen wirksamen Fix anwenden.
4. Das voll relevante Validierungsset ausfuehren.
5. Changelog und Release-Decision-Log aktualisieren.
6. Die Public Surface archivieren, wenn der Patch Release-Policy oder oeffentliche Vertraege aendert.

## Patch-Grenzen

- keine stillen Policy-Aenderungen
- keine unreviewten Release-Automations-Aenderungen
- kein Patch darf neue oeffentliche Claims ohne Evidence hinzufuegen
- Hotfixes brauchen fuer erforderliche Public Docs trotzdem EN- und DE-Parity
