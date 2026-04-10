"""Tests for TuneForge release bundle publishing helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from finetune.model_publisher import (
    BenchmarkSummary,
    EnvironmentManifest,
    LicenseManifest,
    ModelCard,
    TesterAttestation,
    TrainingManifest,
    build_hf_repo_id,
    build_ollama_model_name,
    main,
    write_release_bundle,
)


def test_model_card_generation():
    card = ModelCard(
        model_name="local/tuneforge-sps-plc-qwen3-8b-adapter",
        base_model="Qwen/Qwen3-8B",
        language="de",
        domain="Industrial Automation / IEC 61131-3",
        metrics={"compile_rate": 0.70, "semantic_accuracy": 0.45},
        training_data="Synthetic IEC 61131-3 Structured Text",
        license="MIT",
    )
    readme = card.to_markdown()
    assert "tuneforge-sps-plc-qwen3-8b-adapter" in readme
    assert "IEC 61131-3" in readme
    assert "70%" in readme
    assert "45%" in readme
    assert "Out-of-Scope Use" in readme
    assert "Modelfile" in readme
    assert "Technical Preview" in readme


def test_model_card_yaml_frontmatter():
    card = ModelCard(
        model_name="test/model",
        base_model="base/model",
        language="en",
        domain="Test",
    )
    readme = card.to_markdown()
    assert readme.startswith("---")
    assert "license: MIT" in readme
    assert "language: en" in readme


def test_repo_and_ollama_name_builders():
    assert build_hf_repo_id("local", "Legal/DSGVO", "Qwen/Qwen3-8B") == ("local/tuneforge-legal-dsgvo-qwen3-8b-adapter")
    assert build_ollama_model_name("Legal/DSGVO", "Qwen/Qwen3-8B", "q4_k_m") == (
        "tuneforge:legal-dsgvo-qwen3-8b-q4-k-m"
    )


def test_write_release_bundle(tmp_path):
    card = ModelCard(
        model_name="local/tuneforge-legal-qwen3-8b-adapter",
        base_model="Qwen/Qwen3-8B",
        language="de",
        domain="Legal/DSGVO",
        metrics={"eval_loss": 0.765432},
        training_data="Open legal corpus",
    )
    manifest = TrainingManifest(
        base_model="Qwen/Qwen3-8B",
        dataset="Open legal corpus",
        primary_metric="eval_loss",
        metric_goal="minimize",
        hardware="RTX 3090",
        seed=42,
        peak_vram_mb=12000,
        training_seconds=1800,
        git_sha="abc1234",
        license="MIT",
        limitations=["Human review required."],
        public_status="technical_preview",
        hardware_tier="tier_a_rtx_3090_24gb",
    )
    benchmark = BenchmarkSummary(
        name="Legal/DSGVO",
        dataset="Open legal corpus",
        hardware="RTX 3090",
        primary_metric_name="eval_loss",
        primary_metric_value=0.765432,
        metric_goal="minimize",
        metrics={"eval_loss": 0.765432},
        public_status="technical_preview",
        hardware_tier="tier_a_rtx_3090_24gb",
    )
    license_manifest = LicenseManifest(
        release_license="MIT",
        base_model_license="Apache-2.0",
        upstream_components={"karpathy/autoresearch": "MIT"},
    )
    environment_manifest = EnvironmentManifest(
        hardware_tier="tier_a_rtx_3090_24gb",
        hardware="RTX 3090",
        gpu_model="RTX 3090",
        gpu_vram_gb=24,
        driver_version="551.86",
        cuda_version="12.4",
        os_name="Windows 11",
        docker_image="ghcr.io/ai-engineerings-at/tuneforge-studio:tuneforge-v0.2.0",
        python_version="3.11.9",
    )
    tester_attestation = TesterAttestation(
        tester_id="pilot-3090-a",
        tester_organization="External Pilot",
        submission_kind="private_pilot",
        result_status="passed",
        independent_environment=True,
        submitted_at="2026-03-18T12:00:00Z",
        notes="Successful end-to-end run.",
    )

    paths = write_release_bundle(
        adapter_path=tmp_path,
        card=card,
        manifest=manifest,
        benchmark=benchmark,
        license_manifest=license_manifest,
        environment_manifest=environment_manifest,
        tester_attestation=tester_attestation,
        gguf_filename="model.gguf",
        ollama_model_name="tuneforge:legal-dsgvo-qwen3-8b-q4-k-m",
    )

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "training-manifest.json").exists()
    assert (tmp_path / "benchmark-summary.json").exists()
    assert (tmp_path / "benchmark-summary.md").exists()
    assert (tmp_path / "license-manifest.json").exists()
    assert (tmp_path / "environment-manifest.json").exists()
    assert (tmp_path / "tester-attestation.json").exists()
    assert (tmp_path / "Modelfile").exists()
    assert "ollama create" in (tmp_path / "Modelfile").read_text(encoding="utf-8")
    assert paths["readme"].name == "README.md"


def test_bundle_cli_smoke(tmp_path):
    rc = main(
        [
            "bundle",
            "--adapter-path",
            str(tmp_path),
            "--model-name",
            "local/tuneforge-sps-plc-qwen3-8b-adapter",
            "--base-model",
            "Qwen/Qwen3-8B",
            "--language",
            "de",
            "--domain",
            "Industrial Automation / IEC 61131-3",
            "--training-data",
            "Synthetic IEC data",
            "--dataset",
            "Synthetic IEC data",
            "--primary-metric",
            "eval_loss",
            "--metric-goal",
            "minimize",
            "--primary-value",
            "0.5",
            "--hardware",
            "RTX 3090",
            "--hardware-tier",
            "tier_a_rtx_3090_24gb",
            "--gpu-model",
            "RTX 3090",
            "--gpu-vram-gb",
            "24",
            "--tester-id",
            "ci-smoke",
            "--base-model-license",
            "Apache-2.0",
            "--gguf-filename",
            "model.gguf",
            "--limitation",
            "Human review required.",
        ]
    )
    assert rc == 0
    assert (tmp_path / "training-manifest.json").exists()
    assert (tmp_path / "benchmark-summary.json").exists()
    assert (tmp_path / "environment-manifest.json").exists()
    assert (tmp_path / "tester-attestation.json").exists()
    assert (tmp_path / "Modelfile").exists()
