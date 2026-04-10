#!/usr/bin/env python3
"""
Update validation registry after successful Tier A validation.

Usage:
    python scripts/update_validation_registry.py \
        --tier tier_a_rtx_3090_24gb \
        --evidence-dir validation/evidence/2026-04-10 \
        --tester "Your Name"

This script updates validation/registry.json with the new validation run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except subprocess.CalledProcessError:
        return "unknown"


def load_registry(registry_path: Path) -> dict:
    """Load existing registry."""
    if registry_path.exists():
        return json.loads(registry_path.read_text())
    return {
        "public_status": "technical_preview",
        "runs": [],
        "tiers": {},
        "version": 1,
    }


def save_registry(registry: dict, registry_path: Path) -> None:
    """Save registry with proper formatting."""
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")


def update_tier_status(registry: dict, tier: str) -> None:
    """Update tier status based on successful runs."""
    if tier not in registry["tiers"]:
        print(f"Warning: Tier {tier} not found in registry tiers", file=sys.stderr)
        return

    tier_info = registry["tiers"][tier]
    required_runs = tier_info.get("required_successful_runs", 2)

    # Count successful runs for this tier
    successful_runs = sum(1 for run in registry["runs"] if run.get("tier") == tier and run.get("result") == "success")

    if successful_runs >= required_runs:
        tier_info["status"] = "verified"
        print(f"\n✅ Tier '{tier}' status updated to 'verified' ({successful_runs} runs)")
    else:
        print(f"\n⏳ Tier '{tier}' needs {required_runs - successful_runs} more successful runs")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update validation registry after successful validation")
    parser.add_argument(
        "--tier",
        required=True,
        choices=["tier_a_rtx_3090_24gb", "tier_b_48gb_plus"],
        help="Hardware tier that was validated",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Directory containing validation evidence",
    )
    parser.add_argument(
        "--tester",
        required=True,
        help="Name/ID of the tester",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tier-a-validation.yaml"),
        help="Path to validation config file",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("validation/registry.json"),
        help="Path to registry file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if comparison.json shows failure",
    )

    args = parser.parse_args()

    # Validate evidence directory
    if not args.evidence_dir.exists():
        print(f"Error: Evidence directory not found: {args.evidence_dir}", file=sys.stderr)
        return 1

    # Check comparison.json
    comparison_path = args.evidence_dir / "comparison.json"
    if not comparison_path.exists():
        print(f"Error: comparison.json not found in {args.evidence_dir}", file=sys.stderr)
        print("Run collect_validation_evidence.py first", file=sys.stderr)
        return 1

    comparison = json.loads(comparison_path.read_text())

    if not comparison.get("within_tolerance", False):
        if not args.force:
            print("Error: Validation comparison shows failure", file=sys.stderr)
            print("Use --force to update anyway", file=sys.stderr)
            return 1
        else:
            print("⚠️  Warning: Forcing registry update despite validation failure")

    # Load registry
    registry = load_registry(args.registry)

    # Create run entry
    run_entry = {
        "tier": args.tier,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "tester": args.tester,
        "result": "success" if comparison.get("within_tolerance", False) else "failed",
        "evidence_path": str(args.evidence_dir),
        "git_sha": get_git_sha(),
        "config": str(args.config),
        "metrics": {
            "primary_metric_value": comparison.get("run1", {}).get("primary_metric_value"),
            "eval_loss": comparison.get("run1", {}).get("eval_loss"),
            "peak_vram_mb": comparison.get("run1", {}).get("peak_vram_mb"),
        },
    }

    # Add to registry
    registry["runs"].append(run_entry)

    # Update tier status
    update_tier_status(registry, args.tier)

    # Save registry
    save_registry(registry, args.registry)

    print(f"\n✅ Registry updated: {args.registry}")
    print("\nRun entry added:")
    print(f"  Tier: {args.tier}")
    print(f"  Date: {run_entry['date']}")
    print(f"  Tester: {args.tester}")
    print(f"  Git SHA: {run_entry['git_sha']}")
    print(f"  Result: {run_entry['result']}")

    # Print current status
    print("\nCurrent Registry Status:")
    for tier_id, tier_info in registry["tiers"].items():
        status_icon = "✅" if tier_info.get("status") == "verified" else "⏳"
        print(f"  {status_icon} {tier_id}: {tier_info.get('status', 'unknown')}")

    # Check if we can update public status
    if registry["tiers"].get("tier_a_rtx_3090_24gb", {}).get("status") == "verified":
        print("\n🎉 Tier A is now verified!")
        print("Consider updating README.md to reflect verified status")

    return 0


if __name__ == "__main__":
    sys.exit(main())
