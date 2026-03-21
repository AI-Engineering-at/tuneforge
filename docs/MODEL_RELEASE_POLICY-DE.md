# TuneForge Modell-Release-Policy
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: MODEL_RELEASE_POLICY-EN.md

Der aktuelle Default-Release-Status ist **Technical Preview**.

## Verbindliches Bundle

Jeder oeffentliche Modell-Release muss bereitstellen:

- `README.md`
- `training-manifest.json`
- `benchmark-summary.json`
- `benchmark-summary.md`
- `license-manifest.json`
- `environment-manifest.json`
- `tester-attestation.json`
- `validation-result.json`
- `Modelfile`, wenn GGUF-Packaging erzeugt wird

## Verbindliche Dokumentation

- vollstaendig ausgefuellte Model Card
- Provenance-Record fuer das Dataset
- Benchmark-Evidence-Record
- Release-Approval-Entscheidung
- Risk Classification und Intended-Use-Bewertung
- DPIA- oder VVT-Input, wenn der Deployment-Kontext das verlangt

## Erlaubte Ziele

- GitHub-Release-Artefakte
- Hugging-Face-Adapter-Repositories
- Ollama-kompatibles GGUF-Packaging mit `Modelfile`

## Nicht erlaubte Claims

- keine Sprache fuer Rechts- oder Compliance-Zertifizierung
- kein verifiziertes Hardware-Label ohne passenden Registry-Proof
- kein Benchmark-Sieg ohne gematchtes Hardware-Budget
- keine vagen Qualitaets-Claims ohne angehaengte Evidence

## Review-Ablauf

- Engineering-Review fuer Artefaktintegritaet
- Governance-Review fuer Provenance, Lizenzierung und Dokumentationsvollstaendigkeit
- explizite Entscheidung `keep`, `delay`, `publish` oder `rollback` im Release-Decision-Log

## Oeffentliche Namenskonventionen

- Hugging-Face-Adapter-Konvention: `<org>/tuneforge-<domain>-<base>-adapter`
- Ollama-Konvention: `tuneforge:<domain>-<base>-<quant>`
