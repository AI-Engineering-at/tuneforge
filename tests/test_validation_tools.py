"""Tests for validation registry and public-claim tooling."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scripts.check_public_claims as claims_mod
import scripts.validate_validation_registry as registry_mod


def test_validation_registry_rejects_verified_tier_without_enough_runs():
    payload = {
        "version": 1,
        "public_status": "technical_preview",
        "tiers": {
            "tier_a_rtx_3090_24gb": {
                "label": "Verified on RTX 3090",
                "required_successful_runs": 2,
                "status": "verified",
                "notes": "Primary tier.",
            },
            "tier_b_48gb_plus": {
                "label": "Verified on 48 GB+",
                "required_successful_runs": 2,
                "status": "unverified",
                "notes": "Secondary tier.",
            },
        },
        "runs": [
            {
                "run_id": "pilot-1",
                "tier": "tier_a_rtx_3090_24gb",
                "model": "Qwen/Qwen3-8B",
                "config": "finetune/configs/sps-plc-qwen3.yaml",
                "run_date": "2026-03-18",
                "tester_id": "pilot-a",
                "result_status": "passed",
                "independent_environment": True,
                "artifact_path": "output/pilot-1",
                "notes": "Successful end-to-end run.",
            }
        ],
    }
    errors = registry_mod.validate_registry(payload)
    assert any("only has 1/2 independent successful runs" in error for error in errors)


def test_check_public_claims_respects_preview_registry(tmp_path, monkeypatch):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    registry_path = validation_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "public_status": "technical_preview",
                "tiers": {
                    "tier_a_rtx_3090_24gb": {
                        "label": "Verified on RTX 3090",
                        "required_successful_runs": 2,
                        "status": "unverified",
                        "notes": "Primary tier.",
                    },
                    "tier_b_48gb_plus": {
                        "label": "Verified on 48 GB+",
                        "required_successful_runs": 2,
                        "status": "unverified",
                        "notes": "Secondary tier.",
                    },
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("# TuneForge\n\nTechnical Preview\n", encoding="utf-8")

    monkeypatch.setattr(claims_mod, "ROOT", tmp_path)
    monkeypatch.setattr(claims_mod, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(claims_mod, "iter_public_docs", lambda: [readme])

    assert claims_mod.check_public_claims() == []

    readme.write_text("# TuneForge\n\nTechnical Preview\n\nVerified on RTX 3090\n", encoding="utf-8")
    errors = claims_mod.check_public_claims()
    assert any("Verified on RTX 3090" in error for error in errors)
