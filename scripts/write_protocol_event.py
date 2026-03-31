#!/usr/bin/env python3
"""Append a schema-compatible protocol event to results/protocol.jsonl."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_event(args: argparse.Namespace) -> dict[str, Any]:
    event: dict[str, Any] = {
        "timestamp": args.timestamp or _timestamp(),
        "run_id": args.run_id,
        "event_type": args.event_type,
        "stage": args.stage,
        "status": args.status,
        "message": args.message,
        "git_sha": args.git_sha,
    }
    optional_fields = {
        "hardware_tier": args.hardware_tier,
        "public_status": args.public_status,
        "metric_name": args.metric_name,
        "metric_value": args.metric_value,
        "artifact_path": args.artifact_path,
        "error_id": args.error_id,
    }
    for key, value in optional_fields.items():
        if value not in ("", None):
            event[key] = value
    return event


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a TuneForge protocol event")
    parser.add_argument("--protocol-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--hardware-tier", default="")
    parser.add_argument("--public-status", default="")
    parser.add_argument("--metric-name", default="")
    parser.add_argument("--metric-value", type=float)
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--error-id", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = Path(args.protocol_file)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)

    event = build_event(args)
    with protocol_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    print(protocol_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
