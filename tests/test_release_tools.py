"""Tests for release validation and notebook content rendering tools."""
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_render_content_bundle(tmp_path):
    source = tmp_path / "entry.json"
    source.write_text(json.dumps({
        "title": "TuneForge SPS benchmark",
        "summary": "Qwen3 beat the control on the same RTX 3090 budget.",
        "benchmark": "eval_loss dropped from 0.62 to 0.55.",
        "decision": "Publish candidate adapter.",
    }), encoding="utf-8")

    script = ROOT / "scripts" / "render_content_bundle.py"
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (tmp_path / "out" / "wiki.md").exists()
    assert (tmp_path / "out" / "release-notes.md").exists()


def test_validate_release_artifacts_script(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "README.md").write_text("# Test\n", encoding="utf-8")
    (bundle_dir / "training-manifest.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-8B",
        "dataset": "Synthetic IEC data",
        "primary_metric": "eval_loss",
        "metric_goal": "minimize",
        "hardware": "RTX 3090",
        "seed": 42,
        "peak_vram_mb": 12000,
        "training_seconds": 1800,
        "git_sha": "abc1234",
        "license": "MIT",
        "limitations": ["Human review required."],
    }), encoding="utf-8")
    (bundle_dir / "benchmark-summary.json").write_text(json.dumps({
        "name": "SPS",
        "dataset": "Synthetic IEC data",
        "hardware": "RTX 3090",
        "primary_metric_name": "eval_loss",
        "primary_metric_value": 0.5,
        "metric_goal": "minimize",
        "metrics": {"eval_loss": 0.5},
        "public_status": "technical_preview",
        "hardware_tier": "tier_a_rtx_3090_24gb",
    }), encoding="utf-8")
    (bundle_dir / "benchmark-summary.md").write_text("# Benchmark\n", encoding="utf-8")
    (bundle_dir / "license-manifest.json").write_text(json.dumps({
        "release_license": "MIT",
        "base_model_license": "Apache-2.0",
        "upstream_components": {"karpathy/autoresearch": "MIT"},
    }), encoding="utf-8")
    (bundle_dir / "environment-manifest.json").write_text(json.dumps({
        "hardware_tier": "tier_a_rtx_3090_24gb",
        "hardware": "RTX 3090",
        "gpu_model": "RTX 3090",
        "gpu_vram_gb": 24,
        "driver_version": "551.86",
        "cuda_version": "12.4",
        "os_name": "Windows 11",
        "docker_image": "ghcr.io/ai-engineerings-at/tuneforge-studio:tuneforge-v0.2.0",
        "python_version": "3.11.9",
    }), encoding="utf-8")
    (bundle_dir / "tester-attestation.json").write_text(json.dumps({
        "tester_id": "pilot-3090-a",
        "tester_organization": "External Pilot",
        "submission_kind": "private_pilot",
        "result_status": "passed",
        "independent_environment": True,
        "submitted_at": "2026-03-18T12:00:00Z",
        "notes": "Successful end-to-end run.",
        "artifacts_complete": True,
    }), encoding="utf-8")
    manifest = json.loads((bundle_dir / "training-manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "backend": "peft_trl",
        "version": "0.2.0",
        "public_status": "technical_preview",
        "hardware_tier": "tier_a_rtx_3090_24gb",
    })
    (bundle_dir / "training-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    script = ROOT / "scripts" / "validate_release_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (bundle_dir / "validation-result.json").exists()
