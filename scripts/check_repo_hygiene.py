#!/usr/bin/env python3
"""Validate TuneForge public repo hygiene and tracked artifact rules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRODUCT_ROOT.parent.parent
PRODUCT_PREFIX = "products/tuneforge/"
FORBIDDEN_PREFIXES = (
    "output/",
    "tmp/",
    "logs/",
    "datasets/generated/",
    "results/experiments/",
    "__pycache__/",
    ".pytest_cache/",
    "node_modules/",
)
FORBIDDEN_SUFFIXES = (
    ".env",
    ".gguf",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".log",
)
FORBIDDEN_NAMES = {
    "results/protocol.jsonl",
    "train.py.bak",
}
REQUIRED_FILES = {
    ".env.example",
    "Dockerfile",
    "Dockerfile.finetune",
    "docker-compose.yml",
    "docker-compose.finetune.yml",
}


def _normalize_tracked_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for path in paths:
        if path.startswith(PRODUCT_PREFIX):
            normalized.append(path[len(PRODUCT_PREFIX):])
    return normalized


def get_tracked_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "products/tuneforge",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "git ls-files failed")
    return _normalize_tracked_paths([line.strip() for line in result.stdout.splitlines() if line.strip()])


def check_repo_hygiene(paths: list[str]) -> list[str]:
    errors: list[str] = []
    tracked = set(paths)

    for required in REQUIRED_FILES:
        if required not in tracked:
            errors.append(f"Missing required public repo file: {required}")

    for path in sorted(tracked):
        if path == ".env.example":
            continue
        if path == ".env" or (path.startswith(".env.") and path != ".env.example"):
            errors.append(f"Tracked secret/config file is forbidden: {path}")
        if path in FORBIDDEN_NAMES:
            errors.append(f"Tracked generated/runtime file is forbidden: {path}")
        if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            errors.append(f"Tracked generated/runtime directory is forbidden: {path}")
        if any(path.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"Tracked generated/runtime artifact is forbidden: {path}")

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        tracked = get_tracked_paths()
    except RuntimeError as exc:
        print(f"Repo hygiene check failed: {exc}")
        return 1

    errors = check_repo_hygiene(tracked)
    if errors:
        print("Repo hygiene check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repo hygiene OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
