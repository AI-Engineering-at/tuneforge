# TuneForge Model Documentation SOP
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: MODEL_DOCUMENTATION_SOP-DE.md

TuneForge model documentation is designed for Austria, DSGVO, and EU AI Act-aware engineering teams. It is **not legal advice**.

## Goal

Document every published or review-ready model artifact with enough technical and governance context to support audit, release review, and rollback decisions.

## Required Documentation Set

- model card
- data provenance record
- benchmark evidence record
- release approval record
- risk classification and intended-use record
- DPIA or VVT input when required
- logging and traceability checklist

## Model Card Rules

- identify the exact base model and license
- state the public status and hardware tier
- describe intended use and out-of-scope use
- list limitations, safety notes, and privacy notes
- avoid legal guarantees and unsupported performance claims

## Provenance Rules

- record source systems, access dates, and license posture
- record whether legal-source corpora or Austrian or EU legal references were used
- mark whether the data is public, synthetic, customer-bound, or mixed

## Benchmark Rules

- attach the same primary metric used for promotion decisions
- state hardware, VRAM class, runtime context, and config
- separate preview results from verified hardware labels

## Governance Notes

- Austria and DSGVO review require human-review boundaries to be documented
- EU AI Act review requires intended-use and risk framing to stay versioned and reviewable
- release documentation must remain consistent with `validation/registry.json`
