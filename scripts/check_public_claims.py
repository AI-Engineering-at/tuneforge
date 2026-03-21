#!/usr/bin/env python3
"""Validate public TuneForge docs against the hardware validation registry."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "validation" / "registry.json"
ALWAYS_FORBIDDEN_PATTERNS = (
    re.compile(r"\bproduction-ready\b", re.IGNORECASE),
    re.compile(r"\bproduction ready\b", re.IGNORECASE),
    re.compile(r"\benterprise ready\b", re.IGNORECASE),
    re.compile(r"\bguaranteed compliance\b", re.IGNORECASE),
    re.compile(r"\bcompliance certified\b", re.IGNORECASE),
)
GATED_LABELS = {
    "Verified on RTX 3090": "tier_a_rtx_3090_24gb",
    "Verified on 48 GB+": "tier_b_48gb_plus",
}


def iter_public_docs() -> list[Path]:
    paths = [
        *sorted(ROOT.glob("*.md")),
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "content").glob("*.md")),
        *sorted((ROOT / "templates").glob("*.md")),
        *sorted((ROOT / "validation").glob("*.md")),
    ]
    return [path for path in paths if "archive" not in path.parts]


def _load_registry_statuses() -> dict[str, str]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        tier_name: tier_payload.get("status", "unverified")
        for tier_name, tier_payload in payload.get("tiers", {}).items()
    } | {"public_status": payload.get("public_status", "technical_preview")}


def check_public_claims() -> list[str]:
    errors: list[str] = []
    statuses = _load_registry_statuses()
    public_docs = iter_public_docs()

    for path in public_docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in ALWAYS_FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path.relative_to(ROOT)} contains forbidden claim: {pattern.pattern}")

        for label, tier_name in GATED_LABELS.items():
            if label in text and statuses.get(tier_name) != "verified":
                errors.append(
                    f"{path.relative_to(ROOT)} uses gated label '{label}' without verified registry status"
                )

        if path in {ROOT / "README.md", ROOT / "README-DE.md"} and statuses.get("public_status") == "technical_preview":
            if "Technical Preview" not in text:
                errors.append(
                    f"{path.relative_to(ROOT)} must explicitly say Technical Preview while registry is in preview"
                )

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check_public_claims()
    if errors:
        print("Public claim check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Public claims OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
