"""Tests for agent_loop.py — ResultsParser, ResearchAgent, ExperimentRunner."""

import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_loop import (
    ExperimentResult,
    ResultsParser,
    ResearchAgent,
    ExperimentRunner,
)


# --- ExperimentResult ---


def test_experiment_result_defaults():
    r = ExperimentResult()
    assert r.val_bpb == 0.0
    assert r.primary_metric_name == "val_bpb"
    assert r.primary_metric_value == 0.0
    assert r.success is True
    assert r.status == ""


# --- ResultsParser.parse_output ---


def test_parse_output_success():
    output = """Training complete.
val_bpb: 1.234567
training_seconds: 120.5
total_seconds: 130.0
peak_vram_mb: 8192.0
mfu_percent: 35.2
total_tokens_M: 10.5
num_steps: 1000
num_params_M: 124.0
depth: 12
"""
    result = ResultsParser.parse_output(output)
    assert result.success is True
    assert abs(result.val_bpb - 1.234567) < 1e-6
    assert result.primary_metric_name == "val_bpb"
    assert abs(result.primary_metric_value - 1.234567) < 1e-6
    assert result.training_seconds == 120.5
    assert result.peak_vram_mb == 8192.0
    assert result.num_steps == 1000
    assert result.depth == 12


def test_parse_output_generic_metric():
    output = """Training complete.
primary_metric_name: eval_loss
primary_metric_value: 0.765432
metric_goal: minimize
eval_loss: 0.765432
train_loss: 0.654321
peak_vram_mb: 4096
"""
    result = ResultsParser.parse_output(output)
    assert result.success is True
    assert result.primary_metric_name == "eval_loss"
    assert result.primary_metric_value == 0.765432
    assert result.metric_goal == "minimize"
    assert result.metrics["train_loss"] == 0.654321


def test_parse_output_crash_traceback():
    output = """Loading model...
Traceback (most recent call last):
  File "train.py", line 100
RuntimeError: CUDA out of memory
"""
    result = ResultsParser.parse_output(output)
    assert result.success is False
    assert "CUDA out of memory" in result.error_message


def test_parse_output_crash_killed():
    output = "signal: killed\n"
    result = ResultsParser.parse_output(output)
    assert result.success is False


def test_parse_output_no_val_bpb():
    output = "Training complete.\nnum_steps: 500\n"
    result = ResultsParser.parse_output(output)
    assert result.success is False
    assert "No val_bpb" in result.error_message


# --- ResultsParser.parse_tsv ---


def test_parse_tsv():
    tsv = """commit\tval_bpb\tmemory_gb\tstatus\tdescription
abc1234\t1.500000\t8.0\tkeep\tbaseline
def5678\t1.450000\t8.5\tkeep\tadd rope
ghi9012\t0.000000\t0.0\tcrash\tbroke it
"""
    results = ResultsParser.parse_tsv(tsv)
    assert len(results) == 3
    assert results[0].commit_hash == "abc1234"
    assert results[0].val_bpb == 1.5
    assert results[0].primary_metric_name == "val_bpb"
    assert results[1].status == "keep"
    assert results[2].success is False


def test_parse_tsv_generic_metric():
    tsv = """commit\tmetric_name\tmetric_value\tmemory_gb\tstatus\tdescription
abc1234\teval_loss\t0.500000\t8.0\tkeep\tbaseline
def5678\teval_loss\t0.450000\t8.5\tkeep\timproved
"""
    results = ResultsParser.parse_tsv(tsv)
    assert len(results) == 2
    assert results[0].primary_metric_name == "eval_loss"
    assert results[1].primary_metric_value == 0.45


def test_parse_tsv_empty():
    assert ResultsParser.parse_tsv("") == []
    assert ResultsParser.parse_tsv("header\n") == []


# --- ResultsParser.best_result ---


def test_best_result():
    results = [
        ExperimentResult(val_bpb=1.5, primary_metric_value=1.5, status="keep", success=True),
        ExperimentResult(val_bpb=1.3, primary_metric_value=1.3, status="keep", success=True),
        ExperimentResult(val_bpb=1.1, primary_metric_value=1.1, status="discard", success=True),
        ExperimentResult(val_bpb=0.0, status="crash", success=False),
    ]
    best = ResultsParser.best_result(results)
    assert best.val_bpb == 1.3  # lowest among "keep" status


def test_best_result_maximize():
    results = [
        ExperimentResult(
            primary_metric_name="accuracy",
            primary_metric_value=0.7,
            status="keep",
            success=True,
        ),
        ExperimentResult(
            primary_metric_name="accuracy",
            primary_metric_value=0.8,
            status="keep",
            success=True,
        ),
    ]
    best = ResultsParser.best_result(results, "accuracy", "maximize")
    assert best.primary_metric_value == 0.8


def test_best_result_empty():
    best = ResultsParser.best_result([])
    assert best.val_bpb == 0.0


# --- ResearchAgent ---


def test_research_agent_propose_change():
    mock_provider = MagicMock()
    mock_provider.chat.return_value = """I suggest adding RoPE scaling.

```python
import torch
# modified train.py
print("hello")
```

DESCRIPTION: Add RoPE scaling to attention"""

    agent = ResearchAgent(provider=mock_provider)
    change = agent.propose_change("program text", "original code", [])

    assert change.description == "Add RoPE scaling to attention"
    assert "import torch" in change.code_diff
    assert mock_provider.chat.called


def test_research_agent_no_code_block():
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "Just some text without code"

    agent = ResearchAgent(provider=mock_provider)
    change = agent.propose_change("program", "code", [])

    assert change.code_diff == ""
    assert change.description == "No description"


# --- ExperimentRunner ---


def test_experiment_runner_write_read(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    runner.write_code("print('hello')")
    assert runner.read_code() == "print('hello')"


def test_experiment_runner_append_results_tsv(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    result = ExperimentResult(
        val_bpb=1.234, primary_metric_value=1.234, peak_vram_mb=8192, status="keep", description="test change"
    )
    runner.append_results_tsv("abc1234", result)

    tsv_path = tmp_path / "results" / "results.tsv"
    assert tsv_path.exists()
    content = tsv_path.read_text()
    assert "abc1234" in content
    assert "1.234000" in content
    assert "val_bpb" in content
    assert "keep" in content


def test_experiment_runner_results_tsv_header(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    result = ExperimentResult(val_bpb=1.0, primary_metric_value=1.0, status="keep")
    runner.append_results_tsv("x", result)

    content = (tmp_path / "results" / "results.tsv").read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "commit\tmetric_name\tmetric_value\tmemory_gb\tstatus\tdescription"


def test_experiment_runner_run_training(tmp_path):
    # Write a simple script that outputs val_bpb
    train = tmp_path / "train.py"
    train.write_text("print('val_bpb: 1.500000')")
    runner = ExperimentRunner(work_dir=tmp_path)
    output, returncode = runner.run_training(timeout=30)
    assert "val_bpb: 1.500000" in output
    assert returncode == 0


# --- Hardening Tests ---


def test_experiment_runner_backup_on_write(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    (tmp_path / "train.py").write_text("original")
    runner.write_code("modified")
    assert (tmp_path / "train.py").read_text() == "modified"
    assert (tmp_path / "train.py.bak").read_text() == "original"


def test_experiment_runner_restore_backup(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    (tmp_path / "train.py").write_text("original")
    runner.write_code("modified")
    runner.restore_backup()
    assert (tmp_path / "train.py").read_text() == "original"


def test_experiment_runner_read_missing_file(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.read_code()


def test_experiment_runner_write_json_protocol(tmp_path):
    runner = ExperimentRunner(work_dir=tmp_path)
    result = ExperimentResult(
        val_bpb=1.234,
        primary_metric_value=1.234,
        peak_vram_mb=8192,
        status="keep",
        description="test change",
        commit_hash="abc1234",
    )
    runner.write_experiment_json(1, result)

    import json

    exp_file = tmp_path / "results" / "experiments" / "exp-0001.json"
    assert exp_file.exists()
    data = json.loads(exp_file.read_text())
    assert data["val_bpb"] == 1.234
    assert data["primary_metric_name"] == "val_bpb"
    assert data["experiment"] == 1
    assert "timestamp" in data

    protocol = tmp_path / "results" / "protocol.jsonl"
    assert protocol.exists()
    line = json.loads(protocol.read_text().strip())
    assert line["commit"] == "abc1234"


def test_experiment_runner_gpu_check():
    """GPU check should return a dict (may or may not have GPU)."""
    runner = ExperimentRunner()
    info = runner.check_gpu()
    assert isinstance(info, dict)
    # On CI/dev without GPU, should still return gracefully
    if not info.get("available"):
        assert "error" in info


def test_experiment_runner_training_timeout(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("import time; time.sleep(10)")
    runner = ExperimentRunner(work_dir=tmp_path)
    output, returncode = runner.run_training(timeout=2)
    assert "TIMEOUT" in output
    assert returncode == 1


def test_parse_output_handles_garbage():
    """Parser should not crash on arbitrary output."""
    result = ResultsParser.parse_output("random garbage\nwith lines\nno metrics\n")
    assert result.success is False
    assert "No val_bpb" in result.error_message


def test_research_agent_empty_code_detected():
    """Empty code proposals should be handled."""
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "I don't know what to change."
    agent = ResearchAgent(provider=mock_provider)
    change = agent.propose_change("program", "code", [])
    assert change.code_diff == ""
