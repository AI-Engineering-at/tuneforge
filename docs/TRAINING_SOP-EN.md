# TuneForge Training SOP
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: TRAINING_SOP-DE.md

TuneForge training guidance is written for Austria, DSGVO, and EU AI Act-aware engineering teams. It is **not legal advice**.

## Goal

Provide a repeatable procedure for local fine-tuning and benchmark runs that can be defended in engineering review and later documented for governance.

## Before Training

- choose the control config and the candidate config
- record base model, dataset, license, and intended hardware tier
- verify whether the dataset includes personal data or regulated legal sources
- complete a provenance record before training begins
- decide whether the run is local smoke, private pilot, or release candidate

## Data and Source Discipline

- prefer open or clearly permitted datasets
- record source URLs, access dates, and license terms
- keep legal-source ingestion traceable when using referenced subsystems
- separate public release data from private or customer-specific corpora

## Standard Training Sequence

1. Run tests and CI-equivalent local checks.
2. Run the control benchmark on the target hardware tier.
3. Run the candidate on the same metric and hardware budget.
4. Capture training and evaluation metrics.
5. Generate the release bundle.
6. Validate the release bundle.
7. Update the validation registry only after governance acceptance.

## Required Run Outputs

- primary metric and metric goal
- secondary metrics when available
- training seconds and total seconds
- peak VRAM
- hardware tier
- git SHA
- protocol log path

## Governance Notes

- Austria and DSGVO review require clear provenance and retention notes
- EU AI Act review requires intended-use and risk framing to stay connected to technical evidence
- no public hardware claim is allowed before the validation registry proves it
