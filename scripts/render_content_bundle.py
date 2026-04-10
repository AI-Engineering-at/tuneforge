#!/usr/bin/env python3
"""Render public content assets from a normalized notebook export."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render_bundle(entry: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    title = entry["title"]
    summary = entry["summary"]
    decision = entry.get("decision", "")
    benchmark = entry.get("benchmark", "")
    audience = entry.get("audience", "engineering")

    wiki = f"# {title}\n\n{summary}\n\n## Benchmark\n\n{benchmark}\n\n## Decision\n\n{decision}\n"
    blog = f"# {title}\n\n{summary}\n\n## Why it matters\n\n{entry.get('why_it_matters', benchmark)}\n"
    social = (
        "\n".join(
            [
                title,
                summary,
                f"Audience: {audience}",
                f"Decision: {decision}",
            ]
        )
        + "\n"
    )
    release_notes = (
        "\n".join(
            [
                f"# Release Notes - {title}",
                "",
                summary,
                "",
                "## Benchmark",
                benchmark,
                "",
                "## Release Decision",
                decision,
            ]
        )
        + "\n"
    )

    (output_dir / "wiki.md").write_text(wiki, encoding="utf-8")
    (output_dir / "blog.md").write_text(blog, encoding="utf-8")
    (output_dir / "social.txt").write_text(social, encoding="utf-8")
    (output_dir / "release-notes.md").write_text(release_notes, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print("Usage: render_content_bundle.py <notebook-export.json> <output-dir>")
        return 1

    entry = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    render_bundle(entry, Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
