#!/usr/bin/env python3
"""Validate TuneForge release and validation-run artifact directories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_FILES = {
    "README.md",
    "training-manifest.json",
    "benchmark-summary.json",
    "benchmark-summary.md",
    "license-manifest.json",
    "environment-manifest.json",
    "tester-attestation.json",
}
REQUIRED_MANIFEST_KEYS = {
    "base_model",
    "dataset",
    "primary_metric",
    "metric_goal",
    "hardware",
    "seed",
    "peak_vram_mb",
    "training_seconds",
    "git_sha",
    "license",
    "limitations",
    "backend",
    "version",
    "public_status",
    "hardware_tier",
}
REQUIRED_BENCHMARK_KEYS = {
    "name",
    "dataset",
    "hardware",
    "primary_metric_name",
    "primary_metric_value",
    "metric_goal",
    "metrics",
    "public_status",
    "hardware_tier",
}
REQUIRED_ENVIRONMENT_KEYS = {
    "hardware_tier",
    "hardware",
    "gpu_model",
    "gpu_vram_gb",
    "driver_version",
    "cuda_version",
    "os_name",
    "docker_image",
    "python_version",
}
REQUIRED_ATTESTATION_KEYS = {
    "tester_id",
    "tester_organization",
    "submission_kind",
    "result_status",
    "independent_environment",
    "submitted_at",
    "notes",
    "artifacts_complete",
}
VALID_PUBLIC_STATUSES = {
    "technical_preview",
    "verified_rtx_3090",
    "verified_48gb_plus",
}
VALID_HARDWARE_TIERS = {
    "tier_a_rtx_3090_24gb",
    "tier_b_48gb_plus",
    "unassigned",
}
VALID_RESULT_STATUSES = {"passed", "failed", "partial", "smoke"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_required_keys(payload: dict[str, Any], required_keys: set[str], label: str, errors: list[str], bundle_dir: Path):
    for key in required_keys:
        if key not in payload:
            errors.append(f"{bundle_dir}: {label} missing {key}")


def write_validation_result(bundle_dir: Path, errors: list[str]) -> Path:
    result = {
        "bundle_dir": str(bundle_dir),
        "status": "ok" if not errors else "failed",
        "validated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "validator": "validate_release_artifacts.py",
        "error_count": len(errors),
        "errors": errors,
    }
    path = bundle_dir / "validation-result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_dir(bundle_dir: Path) -> list[str]:
    errors: list[str] = []
    if not bundle_dir.exists():
        return [f"{bundle_dir}: bundle directory does not exist"]
    for filename in REQUIRED_FILES:
        if not (bundle_dir / filename).exists():
            errors.append(f"{bundle_dir}: missing {filename}")

    if errors:
        return errors

    manifest = _read_json(bundle_dir / "training-manifest.json")
    benchmark = _read_json(bundle_dir / "benchmark-summary.json")
    environment = _read_json(bundle_dir / "environment-manifest.json")
    attestation = _read_json(bundle_dir / "tester-attestation.json")

    _has_required_keys(manifest, REQUIRED_MANIFEST_KEYS, "training-manifest.json", errors, bundle_dir)
    _has_required_keys(benchmark, REQUIRED_BENCHMARK_KEYS, "benchmark-summary.json", errors, bundle_dir)
    _has_required_keys(environment, REQUIRED_ENVIRONMENT_KEYS, "environment-manifest.json", errors, bundle_dir)
    _has_required_keys(attestation, REQUIRED_ATTESTATION_KEYS, "tester-attestation.json", errors, bundle_dir)

    if manifest.get("public_status") not in VALID_PUBLIC_STATUSES:
        errors.append(f"{bundle_dir}: invalid public_status {manifest.get('public_status')!r}")
    if manifest.get("hardware_tier") not in VALID_HARDWARE_TIERS:
        errors.append(f"{bundle_dir}: invalid hardware_tier {manifest.get('hardware_tier')!r}")
    if attestation.get("result_status") not in VALID_RESULT_STATUSES:
        errors.append(f"{bundle_dir}: invalid result_status {attestation.get('result_status')!r}")

    if benchmark.get("primary_metric_name") != manifest.get("primary_metric"):
        errors.append(f"{bundle_dir}: primary metric mismatch between benchmark and manifest")
    if benchmark.get("metric_goal") != manifest.get("metric_goal"):
        errors.append(f"{bundle_dir}: metric goal mismatch between benchmark and manifest")
    if benchmark.get("public_status") != manifest.get("public_status"):
        errors.append(f"{bundle_dir}: public_status mismatch between benchmark and manifest")
    if benchmark.get("hardware_tier") != manifest.get("hardware_tier"):
        errors.append(f"{bundle_dir}: hardware_tier mismatch between benchmark and manifest")
    if environment.get("hardware_tier") != manifest.get("hardware_tier"):
        errors.append(f"{bundle_dir}: hardware_tier mismatch between environment and manifest")
    if environment.get("hardware") != manifest.get("hardware"):
        errors.append(f"{bundle_dir}: hardware mismatch between environment and manifest")
    if not attestation.get("artifacts_complete", False):
        errors.append(f"{bundle_dir}: tester attestation marks artifacts as incomplete")

    public_status = manifest.get("public_status")
    hardware_tier = manifest.get("hardware_tier")
    if public_status == "verified_rtx_3090" and hardware_tier != "tier_a_rtx_3090_24gb":
        errors.append(f"{bundle_dir}: verified_rtx_3090 requires tier_a_rtx_3090_24gb")
    if public_status == "verified_48gb_plus" and hardware_tier != "tier_b_48gb_plus":
        errors.append(f"{bundle_dir}: verified_48gb_plus requires tier_b_48gb_plus")
    if public_status != "technical_preview" and attestation.get("result_status") != "passed":
        errors.append(f"{bundle_dir}: verified public_status requires result_status=passed")

    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TuneForge release artifacts")
    parser.add_argument("bundle_dirs", nargs="+", help="Bundle directories to validate")
    parser.add_argument(
        "--skip-write-result",
        action="store_true",
        help="Do not write validation-result.json back into bundle directories",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    all_errors: list[str] = []
    for item in args.bundle_dirs:
        bundle_dir = Path(item)
        errors = validate_dir(bundle_dir)
        if not args.skip_write_result and bundle_dir.exists():
            write_validation_result(bundle_dir, errors)
        all_errors.extend(errors)

    if all_errors:
        print("Release artifact validation failed:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("Release artifacts OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
