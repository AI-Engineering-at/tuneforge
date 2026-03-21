# TuneForge Training SOP
- Language: DE
- Audience: Public
- Last Sync: 2026-03-19
- Pair: TRAINING_SOP-EN.md

Die TuneForge-Trainingsanleitung ist fuer Oesterreich-, DSGVO- und EU-AI-Act-bewusste Engineering-Teams geschrieben. Sie ist **keine Rechtsberatung**.

## Ziel

Ein wiederholbarer Ablauf fuer lokales Fine-Tuning und Benchmark-Runs, der sich in Engineering-Reviews verteidigen und spaeter fuer Governance dokumentieren laesst.

## Vor dem Training

- Control-Config und Kandidaten-Config festlegen
- Basismodell, Dataset, Lizenz und vorgesehenes Hardware-Tier erfassen
- pruefen, ob das Dataset personenbezogene Daten oder regulierte Rechtsquellen enthaelt
- einen Provenance-Record vor Trainingsbeginn ausfuellen
- festlegen, ob der Run `local_smoke`, `private_pilot` oder `release_candidate` ist

## Daten- und Quellen-Disziplin

- offene oder klar erlaubte Datasets bevorzugen
- Source-URLs, Zugriffsdatum und Lizenzbedingungen erfassen
- Legal-Source-Ingestion rueckverfolgbar halten, wenn referenzierte Subsysteme verwendet werden
- oeffentliche Release-Daten von privaten oder kundenspezifischen Korpora trennen

## Standard-Trainingsablauf

1. Tests und lokale CI-aequivalente Checks ausfuehren.
2. Den Control-Benchmark auf dem Ziel-Hardware-Tier ausfuehren.
3. Den Kandidaten auf derselben Metrik und demselben Hardware-Budget ausfuehren.
4. Trainings- und Evaluationsmetriken erfassen.
5. Das Release-Bundle erzeugen.
6. Das Release-Bundle validieren.
7. Die Validation Registry erst nach Governance-Freigabe aktualisieren.

## Verbindliche Run-Outputs

- primaere Metrik und Metric Goal
- sekundaere Metriken, wenn verfuegbar
- Training Seconds und Total Seconds
- Peak VRAM
- Hardware Tier
- Git SHA
- Pfad zum Protocol Log

## Governance-Hinweise

- Oesterreich- und DSGVO-Review brauchen klare Provenance- und Retention-Hinweise
- EU AI Act Review braucht eine Intended-Use- und Risk-Formulierung mit direkter Verbindung zur technischen Evidence
- kein oeffentlicher Hardware-Claim ist erlaubt, bevor die Validation Registry ihn belegt
