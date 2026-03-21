# TuneForge Legal Source References
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: LEGAL_SOURCE_REFERENCES-DE.md

## Purpose

Document the referenced legal-source subsystem used for provenance support. This document describes references only. It does not vendor `legal-scraper` into TuneForge product code.

## Referenced Subsystem

- subsystem: `legal-scraper`
- role: legal source ingestion and provenance support
- phase boundary: referenced subsystem only, not integrated TuneForge runtime code

## Official Source Families Currently Identified

- RIS Austria
- EUR-Lex
- Gesetze-im-Internet
- Fedlex

## Expected Output

- referenced legal corpus inventory
- source URLs and access dates
- provenance support for model documentation
- governance input for Austria and EU legal-source handling

## Non-Goals

- no direct code vendoring into TuneForge in this phase
- no automatic legal-compliance guarantee
- no assumption that scraped legal text is always suitable for public model training without review
