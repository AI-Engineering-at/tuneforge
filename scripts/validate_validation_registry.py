#!/usr/bin/env python3
"""Validate the TuneForge hardware validation registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "validation" / "registry.json"
REQUIRED_TIER_KEYS = {"label", "required_successful_runs", "status", "notes"}
REQUIRED_RUN_KEYS = {
    "run_id",
    "tier",
    "model",
    "config",
    "run_date",
    "tester_id",
    "result_status",
    "independent_environment",
    "artifact_path",
    "notes",
}
REQUIRED_TIERS = {
    "tier_a_rtx_3090_24gb",
    "tier_b_48gb_plus",
}
VALID_TIER_STATUSES = {"unverified", "verified"}
VALID_RESULT_STATUSES = {"passed", "failed", "partial", "smoke"}
VALID_PUBLIC_STATUSES = {
    "technical_preview",
    "verified_rtx_3090",
    "verified_48gb_plus",
}


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("version", "public_status", "tiers", "runs"):
        if key not in payload:
            errors.append(f"registry missing {key}")

    if errors:
        return errors

    if payload["public_status"] not in VALID_PUBLIC_STATUSES:
        errors.append(f"registry has invalid public_status {payload['public_status']!r}")

    tiers = payload["tiers"]
    missing_tiers = REQUIRED_TIERS - set(tiers)
    for tier in sorted(missing_tiers):
        errors.append(f"registry missing tier {tier}")

    for tier_name, tier_payload in tiers.items():
        for key in REQUIRED_TIER_KEYS:
            if key not in tier_payload:
                errors.append(f"tier {tier_name} missing {key}")
        if tier_payload.get("status") not in VALID_TIER_STATUSES:
            errors.append(f"tier {tier_name} has invalid status {tier_payload.get('status')!r}")

    successful_runs_by_tier: dict[str, set[str]] = {tier: set() for tier in REQUIRED_TIERS}
    for run in payload["runs"]:
        for key in REQUIRED_RUN_KEYS:
            if key not in run:
                errors.append(f"run entry missing {key}")
        tier = run.get("tier")
        if tier not in REQUIRED_TIERS:
            errors.append(f"run {run.get('run_id', '<unknown>')} has invalid tier {tier!r}")
            continue
        if run.get("result_status") not in VALID_RESULT_STATUSES:
            errors.append(
                f"run {run.get('run_id', '<unknown>')} has invalid result_status {run.get('result_status')!r}"
            )
        if run.get("result_status") == "passed" and run.get("independent_environment"):
            successful_runs_by_tier[tier].add(run.get("tester_id", "unknown"))

    for tier_name in REQUIRED_TIERS:
        tier_payload = tiers.get(tier_name, {})
        required_runs = int(tier_payload.get("required_successful_runs", 0))
        successful_runs = len(successful_runs_by_tier[tier_name])
        if tier_payload.get("status") == "verified" and successful_runs < required_runs:
            errors.append(
                f"tier {tier_name} is marked verified but only has {successful_runs}/{required_runs} "
                "independent successful runs"
            )

    if payload["public_status"] != "technical_preview":
        not_verified = [
            tier_name for tier_name in REQUIRED_TIERS if tiers.get(tier_name, {}).get("status") != "verified"
        ]
        if not_verified:
            errors.append("registry public_status may only leave technical_preview after both tiers are verified")

    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TuneForge hardware validation registry")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--require-tier-verified",
        action="append",
        default=[],
        help="Fail unless the specified hardware tier is verified in the registry",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    registry_path = Path(args.registry)
    payload = load_registry(registry_path)
    errors = validate_registry(payload)

    for tier_name in args.require_tier_verified:
        tier_payload = payload.get("tiers", {}).get(tier_name, {})
        if tier_payload.get("status") != "verified":
            errors.append(f"required tier is not verified: {tier_name}")

    if errors:
        print("Validation registry check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation registry OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
