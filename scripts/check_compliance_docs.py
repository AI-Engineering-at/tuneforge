#!/usr/bin/env python3
"""Validate required Austria/EU/DSGVO markers in public compliance docs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_RULES = {
    "COMPLIANCE_STATEMENT.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
    "COMPLIANCE_STATEMENT-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
    "docs/COMPLIANCE_PACK-EN.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
    "docs/COMPLIANCE_PACK-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
    "docs/TRAINING_SOP-EN.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
    "docs/TRAINING_SOP-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
    "docs/MODEL_DOCUMENTATION_SOP-EN.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
    "docs/MODEL_DOCUMENTATION_SOP-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
    "templates/AUSTRIA_EU_GOVERNANCE_CHECKLIST-EN.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
    "templates/AUSTRIA_EU_GOVERNANCE_CHECKLIST-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
}


def check_compliance_docs() -> list[str]:
    errors: list[str] = []
    for rel_path, tokens in DOC_RULES.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing compliance doc: {rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"{rel_path} missing token: {token}")
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check_compliance_docs()
    if errors:
        print("Compliance doc check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Compliance docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
