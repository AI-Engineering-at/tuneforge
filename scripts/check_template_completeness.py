#!/usr/bin/env python3
"""Validate required public templates for TuneForge."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN_TEMPLATES = {
    "templates/MODEL_CARD_TEMPLATE-EN.md",
    "templates/DATA_PROVENANCE_TEMPLATE-EN.md",
    "templates/BENCHMARK_EVIDENCE_TEMPLATE-EN.md",
    "templates/RELEASE_APPROVAL_TEMPLATE-EN.md",
    "templates/RISK_CLASSIFICATION_TEMPLATE-EN.md",
    "templates/DPIA_VVT_INPUT_TEMPLATE-EN.md",
    "templates/LOGGING_TRACEABILITY_CHECKLIST-EN.md",
    "templates/AUSTRIA_EU_GOVERNANCE_CHECKLIST-EN.md",
}
DE_TEMPLATES = {path.replace("-EN.md", "-DE.md") for path in EN_TEMPLATES}
REQUIRED_SECTIONS = {
    "EN": ["## Required Fields", "## Completion Notes"],
    "DE": ["## Pflichtfelder", "## Hinweise zur Ausfuellung"],
}


def check_template_completeness() -> list[str]:
    errors: list[str] = []
    for rel_path in sorted(EN_TEMPLATES | DE_TEMPLATES):
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing template: {rel_path}")
            continue

        text = path.read_text(encoding="utf-8")
        language = "EN" if rel_path.endswith("-EN.md") else "DE"
        for section in REQUIRED_SECTIONS[language]:
            if section not in text:
                errors.append(f"{rel_path} missing section: {section}")

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check_template_completeness()
    if errors:
        print("Template completeness check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Template completeness OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
