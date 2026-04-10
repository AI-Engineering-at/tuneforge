#!/usr/bin/env python3
"""Validate required DE/EN documentation pairs for the public TuneForge surface."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA_RE = re.compile(r"^- (Language|Audience|Last Sync|Pair): (.+)$", re.MULTILINE)
DOC_PAIRS: dict[str, str] = {
    "README.md": "README-DE.md",
    "REPO_INDEX.md": "REPO_INDEX-DE.md",
    "CONTRIBUTING.md": "CONTRIBUTING-DE.md",
    "SUPPORT.md": "SUPPORT-DE.md",
    "CHANGELOG.md": "CHANGELOG-DE.md",
    "SECURITY.md": "SECURITY-DE.md",
    "COMPLIANCE_STATEMENT.md": "COMPLIANCE_STATEMENT-DE.md",
    "docs/ARCHITECTURE-EN.md": "docs/ARCHITECTURE-DE.md",
    "docs/VALIDATION_MATRIX-EN.md": "docs/VALIDATION_MATRIX-DE.md",
    "docs/SETUP-EN.md": "docs/SETUP-DE.md",
    "docs/TROUBLESHOOTING-EN.md": "docs/TROUBLESHOOTING-DE.md",
    "docs/GPU_TIERS-EN.md": "docs/GPU_TIERS-DE.md",
    "docs/RELEASE_POLICY-EN.md": "docs/RELEASE_POLICY-DE.md",
    "docs/MODEL_RELEASE_POLICY-EN.md": "docs/MODEL_RELEASE_POLICY-DE.md",
    "docs/COMPLIANCE_PACK-EN.md": "docs/COMPLIANCE_PACK-DE.md",
    "docs/TRAINING_SOP-EN.md": "docs/TRAINING_SOP-DE.md",
    "docs/MODEL_DOCUMENTATION_SOP-EN.md": "docs/MODEL_DOCUMENTATION_SOP-DE.md",
    "docs/UPGRADE_PLAN-EN.md": "docs/UPGRADE_PLAN-DE.md",
    "docs/PATCH_PLAN-EN.md": "docs/PATCH_PLAN-DE.md",
    "docs/LOGGING_AUDIT_PROTOCOL-EN.md": "docs/LOGGING_AUDIT_PROTOCOL-DE.md",
    "docs/LEGAL_SOURCE_REFERENCES-EN.md": "docs/LEGAL_SOURCE_REFERENCES-DE.md",
    "validation/PRIVATE_PILOT_RUNBOOK-EN.md": "validation/PRIVATE_PILOT_RUNBOOK-DE.md",
    "templates/MODEL_CARD_TEMPLATE-EN.md": "templates/MODEL_CARD_TEMPLATE-DE.md",
    "templates/DATA_PROVENANCE_TEMPLATE-EN.md": "templates/DATA_PROVENANCE_TEMPLATE-DE.md",
    "templates/BENCHMARK_EVIDENCE_TEMPLATE-EN.md": "templates/BENCHMARK_EVIDENCE_TEMPLATE-DE.md",
    "templates/RELEASE_APPROVAL_TEMPLATE-EN.md": "templates/RELEASE_APPROVAL_TEMPLATE-DE.md",
    "templates/RISK_CLASSIFICATION_TEMPLATE-EN.md": "templates/RISK_CLASSIFICATION_TEMPLATE-DE.md",
    "templates/DPIA_VVT_INPUT_TEMPLATE-EN.md": "templates/DPIA_VVT_INPUT_TEMPLATE-DE.md",
    "templates/LOGGING_TRACEABILITY_CHECKLIST-EN.md": "templates/LOGGING_TRACEABILITY_CHECKLIST-DE.md",
    "templates/AUSTRIA_EU_GOVERNANCE_CHECKLIST-EN.md": "templates/AUSTRIA_EU_GOVERNANCE_CHECKLIST-DE.md",
}


def _parse_metadata(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    metadata = {match.group(1): match.group(2).strip() for match in METADATA_RE.finditer(content)}
    return metadata


def check_docs_parity() -> list[str]:
    errors: list[str] = []

    for en_rel, de_rel in DOC_PAIRS.items():
        en_path = ROOT / en_rel
        de_path = ROOT / de_rel
        if not en_path.exists():
            errors.append(f"Missing EN doc: {en_rel}")
            continue
        if not de_path.exists():
            errors.append(f"Missing DE doc: {de_rel}")
            continue

        en_meta = _parse_metadata(en_path)
        de_meta = _parse_metadata(de_path)

        for label in ("Language", "Audience", "Last Sync", "Pair"):
            if label not in en_meta:
                errors.append(f"{en_rel} missing metadata field: {label}")
            if label not in de_meta:
                errors.append(f"{de_rel} missing metadata field: {label}")

        if en_meta.get("Language") != "EN":
            errors.append(f"{en_rel} must declare Language: EN")
        if de_meta.get("Language") != "DE":
            errors.append(f"{de_rel} must declare Language: DE")
        if en_meta.get("Pair") != de_path.name:
            errors.append(f"{en_rel} must declare Pair: {de_path.name}")
        if de_meta.get("Pair") != en_path.name:
            errors.append(f"{de_rel} must declare Pair: {en_path.name}")
        if en_meta.get("Last Sync") != de_meta.get("Last Sync"):
            errors.append(f"{en_rel} and {de_rel} must share the same Last Sync date")

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check_docs_parity()
    if errors:
        print("Docs parity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Docs parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
