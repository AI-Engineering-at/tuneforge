#!/usr/bin/env python3
"""Verify required TuneForge attribution and governance files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required_files = {
        "README.md": ["TuneForge", "karpathy/autoresearch", "THIRD_PARTY.md"],
        "FORK.md": ["karpathy/autoresearch", "MIT"],
        "THIRD_PARTY.md": ["transformers", "trl", "peft", "unsloth"],
        "SECURITY.md": ["TuneForge", "GHCR"],
        "docs/CREDITS.md": ["Andrej Karpathy"],
    }

    failures: list[str] = []
    for rel_path, tokens in required_files.items():
        path = ROOT / rel_path
        if not path.exists():
            failures.append(f"Missing file: {rel_path}")
            continue
        content = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in content:
                failures.append(f"{rel_path} missing token: {token}")

    if failures:
        print("Attribution check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Attribution OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
