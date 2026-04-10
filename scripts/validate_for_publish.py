#!/usr/bin/env python3
"""
Validate model before publishing to HuggingFace Hub.

Usage:
    python scripts/validate_for_publish.py --model-path ./outputs/my-model --template templates/model_card.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_model_files(model_path: str) -> list[str]:
    """Check that required model files exist."""
    path = Path(model_path)
    errors = []
    
    if not path.exists():
        errors.append(f"Model path does not exist: {path}")
        return errors
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ]
    
    for file in required_files:
        # Check for sharded models (model-00001-of-00002.safetensors)
        if file == "model.safetensors":
            if not list(path.glob("model*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
                errors.append("No model weights found (expected model*.safetensors or pytorch_model*.bin)")
        elif not (path / file).exists():
            errors.append(f"Missing required file: {file}")
    
    return errors


def check_model_card(template_path: str | None) -> list[str]:
    """Check that model card template exists and is valid."""
    errors = []
    
    if template_path and not Path(template_path).exists():
        errors.append(f"Model card template not found: {template_path}")
    
    return errors


def check_validation_report(report_path: str | None) -> list[str]:
    """Check validation report if provided."""
    errors = []
    warnings = []
    
    if not report_path:
        warnings.append("No validation report provided - model will be marked as unverified")
        return errors
    
    path = Path(report_path)
    if not path.exists():
        errors.append(f"Validation report not found: {report_path}")
        return errors
    
    try:
        report = json.loads(path.read_text())
        
        if report.get("status") != "PASS":
            errors.append(f"Validation report shows status: {report.get('status')}")
        
        if report.get("tier") != "A":
            warnings.append(f"Validation tier is {report.get('tier', 'unknown')}, not Tier A")
        
        runs = report.get("verification_runs", [])
        if len(runs) < 2:
            warnings.append(f"Only {len(runs)} verification run(s) - 2 required for Tier A")
        
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in validation report: {report_path}")
    
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model before publishing")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--template", help="Path to model card template")
    parser.add_argument("--validation-report", help="Path to validation report")
    
    args = parser.parse_args()
    
    print(f"Validating model: {args.model_path}")
    print("=" * 60)
    
    all_errors = []
    all_warnings = []
    
    # Check model files
    errors = check_model_files(args.model_path)
    if errors:
        print("❌ Model files issues:")
        for e in errors:
            print(f"  - {e}")
        all_errors.extend(errors)
    else:
        print("✅ Model files valid")
    
    # Check model card
    errors = check_model_card(args.template)
    if errors:
        print("❌ Model card issues:")
        for e in errors:
            print(f"  - {e}")
        all_errors.extend(errors)
    else:
        print("✅ Model card valid")
    
    # Check validation report
    errors = check_validation_report(args.validation_report)
    if errors:
        print("❌ Validation report issues:")
        for e in errors:
            print(f"  - {e}")
        all_errors.extend(errors)
    else:
        print("✅ Validation report valid (or not provided)")
    
    print("=" * 60)
    
    if all_errors:
        print(f"❌ Validation FAILED with {len(all_errors)} error(s)")
        print("Fix errors before publishing.")
        return 1
    else:
        print("✅ Validation PASSED")
        if all_warnings:
            print(f"\nWarnings ({len(all_warnings)}):")
            for w in all_warnings:
                print(f"  - {w}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
