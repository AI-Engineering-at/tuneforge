#!/usr/bin/env python3
"""
Collect and compare validation run evidence.

Usage:
    python scripts/collect_validation_evidence.py run1_dir run2_dir output_dir

Example:
    python scripts/collect_validation_evidence.py \
        output/validation-run-20260410-120000 \
        output/validation-run-20260410-121500 \
        validation/evidence/2026-04-10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def extract_metrics_from_log(log_file: Path) -> dict[str, Any]:
    """Extract key metrics from a training log file."""
    metrics = {
        "primary_metric_value": None,
        "eval_loss": None,
        "peak_vram_mb": None,
        "training_seconds": None,
        "total_seconds": None,
    }
    
    if not log_file.exists():
        return metrics
    
    content = log_file.read_text()
    
    # Extract primary_metric_value
    if match := re.search(r"primary_metric_value:\s+([\d.]+)", content):
        metrics["primary_metric_value"] = float(match.group(1))
    
    # Extract eval_loss
    if match := re.search(r"eval_loss:\s+([\d.]+)", content):
        metrics["eval_loss"] = float(match.group(1))
    
    # Extract peak_vram_mb
    if match := re.search(r"peak_vram_mb:\s+([\d.]+)", content):
        metrics["peak_vram_mb"] = float(match.group(1))
    
    # Extract training_seconds
    if match := re.search(r"training_seconds:\s+([\d.]+)", content):
        metrics["training_seconds"] = float(match.group(1))
    
    # Extract total_seconds
    if match := re.search(r"total_seconds:\s+([\d.]+)", content):
        metrics["total_seconds"] = float(match.group(1))
    
    return metrics


def calculate_difference(run1: dict, run2: dict) -> dict[str, Any]:
    """Calculate percentage difference between two runs."""
    differences = {}
    
    for key in set(run1.keys()) & set(run2.keys()):
        val1 = run1[key]
        val2 = run2[key]
        
        if val1 is None or val2 is None:
            differences[key] = None
            continue
        
        if val1 == 0:
            differences[key] = abs(val2 - val1) * 100
        else:
            differences[key] = abs(val2 - val1) / val1 * 100
    
    return differences


def check_within_tolerance(differences: dict, tolerance: float = 2.0) -> bool:
    """Check if all differences are within tolerance."""
    for key, diff in differences.items():
        if diff is None:
            continue
        if diff > tolerance:
            print(f"FAIL: {key} difference ({diff:.2f}%) exceeds tolerance ({tolerance}%)")
            return False
    
    return True


def copy_artifacts(run_dir: Path, output_dir: Path) -> None:
    """Copy relevant artifacts from run directory."""
    artifacts = [
        "training-manifest.json",
        "benchmark-summary.json",
        "model-card.md",
        "environment-manifest.json",
    ]
    
    for artifact in artifacts:
        source = run_dir / artifact
        if source.exists():
            target = output_dir / artifact
            target.write_bytes(source.read_bytes())
            print(f"  Copied: {artifact}")


def generate_comparison_report(
    run1_metrics: dict,
    run2_metrics: dict,
    differences: dict,
    within_tolerance: bool,
    output_dir: Path,
) -> None:
    """Generate a human-readable comparison report."""
    report_lines = [
        "# Tier A Validation Run Comparison Report",
        "",
        f"**Date:** {output_dir.name}",
        f"**Status:** {'✅ PASSED' if within_tolerance else '❌ FAILED'}",
        "",
        "## Run 1 Metrics",
        "",
    ]
    
    for key, value in run1_metrics.items():
        if value is not None:
            report_lines.append(f"- **{key}:** {value:.6f}")
    
    report_lines.extend([
        "",
        "## Run 2 Metrics",
        "",
    ])
    
    for key, value in run2_metrics.items():
        if value is not None:
            report_lines.append(f"- **{key}:** {value:.6f}")
    
    report_lines.extend([
        "",
        "## Differences (%)",
        "",
        "| Metric | Difference | Status |",
        "|--------|------------|--------|",
    ])
    
    for key, diff in differences.items():
        if diff is not None:
            status = "✅" if diff <= 2.0 else "❌"
            report_lines.append(f"| {key} | {diff:.4f}% | {status} |")
        else:
            report_lines.append(f"| {key} | N/A | ⚠️ |")
    
    report_lines.extend([
        "",
        "## Conclusion",
        "",
    ])
    
    if within_tolerance:
        report_lines.append("Both runs are within the ±2% tolerance. Tier A validation **PASSED**.")
    else:
        report_lines.append("Runs differ by more than ±2%. Tier A validation **FAILED**.")
        report_lines.append("")
        report_lines.append("**Recommendations:**")
        report_lines.append("- Verify seed is set correctly (42)")
        report_lines.append("- Check for other processes using GPU")
        report_lines.append("- Verify deterministic operations are enabled")
    
    report_path = output_dir / "COMPARISON_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nGenerated: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect and compare validation run evidence"
    )
    parser.add_argument("run1_dir", type=Path, help="Directory of first run")
    parser.add_argument("run2_dir", type=Path, help="Directory of second run")
    parser.add_argument("output_dir", type=Path, help="Output directory for evidence")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Tolerance percentage for comparison (default: 2.0)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.run1_dir.exists():
        print(f"Error: Run 1 directory not found: {args.run1_dir}", file=sys.stderr)
        return 1
    
    if not args.run2_dir.exists():
        print(f"Error: Run 2 directory not found: {args.run2_dir}", file=sys.stderr)
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run 1: {args.run1_dir}")
    print(f"Run 2: {args.run2_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Extract metrics from logs
    print("Extracting metrics...")
    run1_log = args.run1_dir / "training.log"
    run2_log = args.run2_dir / "training.log"
    
    # Try to find log file
    if not run1_log.exists():
        # Look for any .log file
        logs = list(args.run1_dir.glob("*.log"))
        if logs:
            run1_log = logs[0]
    
    if not run2_log.exists():
        logs = list(args.run2_dir.glob("*.log"))
        if logs:
            run2_log = logs[0]
    
    run1_metrics = extract_metrics_from_log(run1_log)
    run2_metrics = extract_metrics_from_log(run2_log)
    
    # Calculate differences
    differences = calculate_difference(run1_metrics, run2_metrics)
    
    # Check tolerance
    within_tolerance = check_within_tolerance(differences, args.tolerance)
    
    # Create comparison JSON
    comparison = {
        "run1": run1_metrics,
        "run2": run2_metrics,
        "differences_percent": differences,
        "tolerance_percent": args.tolerance,
        "within_tolerance": within_tolerance,
        "status": "passed" if within_tolerance else "failed",
    }
    
    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"Saved: {comparison_path}")
    
    # Copy artifacts from run 1 (primary)
    print("\nCopying artifacts from run 1...")
    copy_artifacts(args.run1_dir, args.output_dir)
    
    # Generate comparison report
    generate_comparison_report(
        run1_metrics,
        run2_metrics,
        differences,
        within_tolerance,
        args.output_dir,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tolerance: ±{args.tolerance}%")
    print()
    
    for key, diff in differences.items():
        if diff is not None:
            status = "✅" if diff <= args.tolerance else "❌"
            print(f"{status} {key}: {diff:.4f}% difference")
    
    print()
    if within_tolerance:
        print("✅ VALIDATION PASSED")
        print(f"\nEvidence saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Review evidence package")
        print("2. Update validation/registry.json")
        print("3. Create PR with evidence")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("\nRecommendations:")
        print("- Check seed setting (must be 42)")
        print("- Verify deterministic CUDA operations")
        print("- Check for GPU temperature throttling")
        return 1


if __name__ == "__main__":
    sys.exit(main())
