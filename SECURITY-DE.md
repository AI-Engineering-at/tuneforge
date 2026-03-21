# TuneForge Sicherheitsrichtlinie
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: SECURITY.md

TuneForge ist ein oeffentliches Engineering-Framework mit Docker-basierter Produktoberflaeche. Der Sicherheitsprozess muss mit einem spaeteren dedizierten GitHub-Repo und mit GHCR-basierten Releases kompatibel bleiben.

## Geltungsbereich

Diese Richtlinie deckt die oeffentliche TuneForge-Oberflaeche in `products/tuneforge` ab, einschliesslich:

- Quellcode
- Dockerfiles und Compose-Dateien
- GitHub-Actions-Workflows
- Release-Bundle-Werkzeuge
- oeffentliche Dokumentation und Templates
- Validation Registry und Audit-Artefakte

Nicht abgedeckt sind private Infrastruktur, private open-notebook-Inhalte, Kundenumgebungen oder Drittservices ausserhalb unserer Kontrolle.

## Unterstuetzte Oberflaeche

- aktueller `main`
- letzter getaggter TuneForge-Code-Release
- letzter getaggter TuneForge-Studio-Docker-Preview-Release auf GHCR

Der aktuelle oeffentliche Status bleibt **Technical Preview**.

## Meldung von Sicherheitsproblemen

Sicherheitsprobleme muessen privat vor einer oeffentlichen Offenlegung gemeldet werden.

- betroffene Datei, Workflow oder Befehl angeben
- Reproduktionsschritte und erwarteten Impact angeben
- nennen, ob Secrets, Provenance-Daten oder Release-Integritaet betroffen sind
- keine oeffentliche Issue fuer Live-Credentials, Signing- oder Supply-Chain-Probleme anlegen

## Regeln fuer das Public Repo

- keine gebauten Docker-Images in Git
- keine echten `.env`-Dateien oder Secrets in Git
- keine generierten Modellartefakte in Git
- keine nicht belegbaren Security- oder Compliance-Claims
- GHCR-Publikation nur ueber GitHub Actions, nicht ueber committete Artefakte

## Erwartungen an Release-Haertung

Oeffentliche TuneForge-Releases sollen enthalten:

- gruene CI- und Public-Repo-Readiness-Checks
- validierte Release-Artefakte
- gepruefte Attribution und Lizenzierung
- SBOM-Erzeugung fuer GHCR-Docker-Releases
- Checksummen und signierte Metadaten fuer GHCR-Release-Ausgaben

## Grenzen

- TuneForge-Dokumentation ist Governance- und Engineering-Unterstuetzung
- sie ist keine Rechtsberatung
- sie ist keine Sicherheitszertifizierung
- sie garantiert nicht automatisch Compliance fuer Oesterreich, DSGVO oder den EU AI Act
