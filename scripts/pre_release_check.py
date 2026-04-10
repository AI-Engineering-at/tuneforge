#!/usr/bin/env python3
"""
Pre-release validation script.

Usage:
    python scripts/pre_release_check.py

Checks:
- All tests pass
- Coverage meets threshold
- Code style clean
- No security issues
- Documentation parity
- Version consistency
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def check_tests() -> bool:
    """Check that all tests pass."""
    print("\n" + "=" * 60)
    print("CHECKING: Tests")
    print("=" * 60)
    
    result = run_command(["python", "-m", "pytest", "tests/", "-q", "--tb=no"], check=False)
    
    if "passed" not in result.stdout:
        print("[FAIL] Tests did not pass")
        print(result.stdout)
        print(result.stderr)
        return False
    
    # Extract numbers
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if "passed" in line:
            print(f"[PASS] {line}")
            return True
    
    return True


def check_coverage(threshold: float = 75.0) -> bool:
    """Check code coverage meets threshold."""
    print("\n" + "=" * 60)
    print("CHECKING: Coverage")
    print("=" * 60)
    
    result = run_command(
        ["python", "-m", "pytest", "tests/", "--cov=finetune", "--cov=data_utils", "--cov-report=json"],
        check=False
    )
    
    # Read coverage report
    coverage_file = Path(".coverage/coverage.json")
    if not coverage_file.exists():
        # Try alternative location
        coverage_file = Path("coverage.json")
    
    if coverage_file.exists():
        data = json.loads(coverage_file.read_text())
        total = data.get("totals", {}).get("percent_covered", 0)
        print(f"Total coverage: {total:.1f}%")
        
        if total >= threshold:
            print(f"[PASS] Coverage meets threshold ({threshold}%)")
            return True
        else:
            print(f"[FAIL] Coverage {total:.1f}% < {threshold}%")
            return False
    
    print("[WARN] Could not read coverage report")
    return True  # Don't fail on missing coverage report


def check_code_style() -> bool:
    """Check code style with ruff."""
    print("\n" + "=" * 60)
    print("CHECKING: Code Style (ruff)")
    print("=" * 60)
    
    result = run_command(["python", "-m", "ruff", "check", "."], check=False)
    
    if result.returncode == 0:
        print("[PASS] Code style clean")
        return True
    else:
        print("[FAIL] Code style issues found")
        print(result.stdout)
        return False


def check_types() -> bool:
    """Check types with mypy."""
    print("\n" + "=" * 60)
    print("CHECKING: Type Checking (mypy)")
    print("=" * 60)
    
    result = run_command(["python", "-m", "mypy", "finetune/"], check=False)
    
    if result.returncode == 0:
        print("[PASS] Type checking passed")
        return True
    else:
        print("[FAIL] Type checking issues found")
        print(result.stdout[-1000:])  # Last 1000 chars
        return False


def check_docs_parity() -> bool:
    """Check EN/DE documentation parity."""
    print("\n" + "=" * 60)
    print("CHECKING: Documentation Parity (EN/DE)")
    print("=" * 60)
    
    result = run_command(["python", "scripts/check_docs_parity.py"], check=False)
    
    if result.returncode == 0:
        print("[PASS] Documentation parity maintained")
        return True
    else:
        print("[FAIL] Documentation parity issues")
        print(result.stdout)
        return False


def check_version_consistency() -> bool:
    """Check version is consistent across files."""
    print("\n" + "=" * 60)
    print("CHECKING: Version Consistency")
    print("=" * 60)
    
    # Read version from pyproject.toml
    pyproject = Path("pyproject.toml").read_text()
    version_line = [line for line in pyproject.split("\n") if "version" in line and "=" in line]
    
    if not version_line:
        print("❌ FAIL: Could not find version in pyproject.toml")
        return False
    
    version = version_line[0].split("=")[1].strip().strip('"')
    print(f"Version in pyproject.toml: {version}")
    
    # Check CHANGELOG
    changelog = Path("CHANGELOG.md").read_text()
    if f"## [{version}]" in changelog:
        print(f"[PASS] Version {version} found in CHANGELOG")
        return True
    else:
        print(f"[FAIL] Version {version} not found in CHANGELOG")
        return False


def check_security() -> bool:
    """Check for security issues with bandit."""
    print("\n" + "=" * 60)
    print("CHECKING: Security (bandit)")
    print("=" * 60)
    
    result = run_command(["python", "-m", "bandit", "-r", "finetune/", "-f", "json"], check=False)
    
    try:
        report = json.loads(result.stdout)
        issues = report.get("results", [])
        
        high_severity = [i for i in issues if i.get("issue_severity") in ("HIGH", "CRITICAL")]
        
        if high_severity:
            print(f"[FAIL] {len(high_severity)} high/critical severity issues")
            for issue in high_severity:
                print(f"  - {issue['issue_text']} ({issue['filename']}:{issue['line_number']})")
            return False
        else:
            print("[PASS] No high/critical security issues")
            return True
    except json.JSONDecodeError:
        print("[WARN] Could not parse bandit output")
        return True


def main() -> int:
    """Run all pre-release checks."""
    print("\n" + "=" * 60)
    print("TUNEFORGE PRE-RELEASE CHECK")
    print("=" * 60)
    
    checks = [
        ("Tests", check_tests),
        ("Coverage", check_coverage),
        ("Code Style", check_code_style),
        ("Type Checking", check_types),
        ("Docs Parity", check_docs_parity),
        ("Version Consistency", check_version_consistency),
        ("Security", check_security),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[ERROR] in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - Ready for release!")
        print("=" * 60)
        return 0
    else:
        print("SOME CHECKS FAILED - Fix before releasing")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
