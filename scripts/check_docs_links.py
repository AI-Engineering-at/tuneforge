#!/usr/bin/env python3
"""Check relative markdown links inside the TuneForge product surface."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = [
    *ROOT.glob("*.md"),
    *ROOT.glob("docs/*.md"),
    *ROOT.glob("content/*.md"),
    *ROOT.glob("templates/*.md"),
    *ROOT.glob("validation/*.md"),
    *ROOT.glob("validation/templates/*.md"),
]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def should_ignore(target: str) -> bool:
    return (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("#")
    )


def main() -> int:
    missing: list[str] = []
    for md_file in MD_FILES:
        text = md_file.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            target = target.strip()
            if should_ignore(target):
                continue
            clean_target = target.split("#", 1)[0]
            resolved = (md_file.parent / clean_target).resolve()
            if not resolved.exists():
                missing.append(f"{md_file.relative_to(ROOT)} -> {target}")

    if missing:
        print("Missing markdown link targets:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Markdown links OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
