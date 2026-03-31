"""Tests for remote RTX 3090 pilot helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_remote_3090_pilot import (
    RemoteConfig,
    build_legal_pilot_script,
    build_preflight_script,
    build_repo_bootstrap_script,
    encode_powershell,
)


def test_encode_powershell_returns_ascii_payload():
    encoded = encode_powershell("Write-Output 'ok'")
    assert isinstance(encoded, str)
    assert encoded
    assert all(ord(ch) < 128 for ch in encoded)


def test_bootstrap_script_targets_default_windows_repo():
    script = build_repo_bootstrap_script(RemoteConfig())
    assert "C:\\Users\\Joe\\Documents\\tuneforge" in script
    assert "git clone" in script
    assert "git -C $repoDir fetch origin" in script


def test_preflight_script_writes_results_json():
    script = build_preflight_script(RemoteConfig(run_id="preflight-1"))
    assert "results\\' + $runId + '-preflight.json" in script
    assert "nvidia-smi --query-gpu=name,memory.total,driver_version" in script


def test_legal_pilot_script_contains_protocol_and_training_steps():
    script = build_legal_pilot_script(RemoteConfig(run_id="pilot-1"))
    assert "write_protocol_event.py" in script
    assert "download-openlegaldata" in script
    assert "python3 -m finetune.trainer" in script
    assert "validate_release_artifacts.py" in script
