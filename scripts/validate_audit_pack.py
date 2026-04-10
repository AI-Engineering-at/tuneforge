#!/usr/bin/env python3
"""Validate TuneForge audit/logging schema examples and public audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "audit"
REQUIRED_FILES = {
    AUDIT_ROOT / "protocol.schema.json",
    AUDIT_ROOT / "siem-export.schema.json",
    AUDIT_ROOT / "examples" / "protocol.sample.jsonl",
    AUDIT_ROOT / "examples" / "error-registry.sample.json",
    AUDIT_ROOT / "examples" / "incident-log.sample.json",
    AUDIT_ROOT / "examples" / "release-decision-log.sample.json",
    AUDIT_ROOT / "examples" / "siem-event.sample.json",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_required_keys(payload, required_keys: list[str], label: str, errors: list[str]):
    for key in required_keys:
        if key not in payload:
            errors.append(f"{label} missing key: {key}")


def validate_audit_pack() -> list[str]:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing audit artifact: {path.relative_to(ROOT)}")
    if errors:
        return errors

    protocol_schema = _load_json(AUDIT_ROOT / "protocol.schema.json")
    siem_schema = _load_json(AUDIT_ROOT / "siem-export.schema.json")

    protocol_lines = [
        json.loads(line)
        for line in (AUDIT_ROOT / "examples" / "protocol.sample.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, payload in enumerate(protocol_lines, start=1):
        _validate_required_keys(
            payload, protocol_schema.get("required", []), f"protocol.sample.jsonl line {index}", errors
        )

    for rel_name, required in {
        "error-registry.sample.json": ["error_id", "severity", "summary", "mitigation", "last_reviewed"],
        "incident-log.sample.json": ["incident_id", "severity", "opened_at", "status", "summary", "impact"],
        "release-decision-log.sample.json": ["release_id", "decision", "status", "approved_by", "evidence_refs"],
        "siem-event.sample.json": siem_schema.get("required", []),
    }.items():
        payload = _load_json(AUDIT_ROOT / "examples" / rel_name)
        if isinstance(payload, list):
            if not payload:
                errors.append(f"{rel_name} must contain at least one record")
                continue
            _validate_required_keys(payload[0], required, rel_name, errors)
        else:
            _validate_required_keys(payload, required, rel_name, errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = validate_audit_pack()
    if errors:
        print("Audit pack validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Audit pack OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
