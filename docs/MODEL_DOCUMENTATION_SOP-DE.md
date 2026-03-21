# TuneForge Modell-Dokumentations-SOP
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: MODEL_DOCUMENTATION_SOP-EN.md

Die TuneForge-Modell-Dokumentation ist fuer Oesterreich-, DSGVO- und EU-AI-Act-bewusste Engineering-Teams konzipiert. Sie ist **keine Rechtsberatung**.

## Ziel

Jedes veroeffentlichte oder review-bereite Modellartefakt mit ausreichend technischem und Governance-Kontext dokumentieren, um Audit, Release-Review und Rollback-Entscheidungen zu unterstuetzen.

## Verbindlicher Dokumentationssatz

- Model Card
- Data-Provenance-Record
- Benchmark-Evidence-Record
- Release-Approval-Record
- Risk Classification und Intended-Use-Record
- DPIA- oder VVT-Input, wenn erforderlich
- Logging- und Traceability-Checkliste

## Regeln fuer die Model Card

- exaktes Basismodell und Lizenz nennen
- Public Status und Hardware Tier nennen
- Intended Use und Out-of-Scope Use beschreiben
- Limitationen, Safety Notes und Privacy Notes auffuehren
- keine Rechtsgarantien und keine unbelegten Performance-Claims machen

## Provenance-Regeln

- Source-Systeme, Zugriffsdatum und Lizenzstatus erfassen
- markieren, ob Legal-Source-Korpora oder oesterreichische bzw. EU-Rechtsreferenzen verwendet wurden
- kennzeichnen, ob die Daten public, synthetic, customer-bound oder mixed sind

## Benchmark-Regeln

- dieselbe primaere Metrik anhaengen, die fuer Promotionsentscheidungen verwendet wurde
- Hardware, VRAM-Klasse, Laufzeitkontext und Config angeben
- Preview-Ergebnisse von verifizierten Hardware-Labels trennen

## Governance-Hinweise

- Oesterreich- und DSGVO-Review brauchen dokumentierte Human-Review-Grenzen
- EU AI Act Review braucht versionierte und reviewbare Intended-Use- und Risk-Formulierungen
- Release-Dokumentation muss mit `validation/registry.json` konsistent bleiben
