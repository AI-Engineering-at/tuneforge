# TuneForge Release Policy
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: RELEASE_POLICY-EN.md

Der aktuelle Produktstatus ist **Technical Preview**.

## Release-Typen

### Code-Release

- getaggter Git-Commit
- Changelog-Eintrag
- gruene CI
- Public-Repo-Readiness-Checks
- darf veroeffentlicht werden, solange das Produkt Preview bleibt

### Docker-Preview-Release

- kanonisches Image-Ziel: `ghcr.io/ai-engineerings-at/tuneforge-studio:<semver>`
- Build in GitHub Actions
- GHCR-Publikation nur ueber Workflow-Automation
- SBOM, Checksummen und signierte Metadaten erforderlich

### Modell-Release

- vollstaendiges Release-Bundle
- zur Validation Registry kompatibler Status
- vollstaendige Model Card und Provenance-Dokumente
- explizite Freigabeentscheidung dokumentiert
- kein verifiziertes Hardware-Label ohne Registry-Proof

## Verbindliche Gates

- Code-Quality-Gate
- Docs-Parity-Gate
- Repo-Hygiene-Gate
- Compliance-Docs-Gate
- Template-Completeness-Gate
- Audit-Pack-Gate
- Validation-Registry-Gate
- Public-Claim-Gate

## Benchmark-First-Regel

- Controls bleiben Default, bis ein Kandidat auf gleichem Hardware-Budget und gleicher Metrik gewinnt
- veroeffentlichte Vergleiche muessen Hardware, Dataset, Laufzeitkontext und Public Status enthalten
- unveroeffentlichte oder fehlgeschlagene Runs duerfen intern lernen helfen, aber nicht fuer oeffentliche Claims verwendet werden

## Grenzen fuer das Public Repo

- keine gebauten Images in Git
- keine Secrets in Git
- keine generierten Modellartefakte in Git
- keine Sprache, die Compliance garantiert
- keine verifizierten Hardware-Labels ohne Evidence

## Change Management

- strukturelle Aenderungen an der Public Surface brauchen einen Archive-Snapshot
- brechende Aenderungen am Release-Prozess brauchen Migrationshinweise
- Notfall-Fixes folgen [PATCH_PLAN-DE.md](PATCH_PLAN-DE.md)
- strategische Plattformaenderungen folgen [UPGRADE_PLAN-DE.md](UPGRADE_PLAN-DE.md)
