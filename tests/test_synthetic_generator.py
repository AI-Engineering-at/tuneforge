"""Tests for synthetic data generator and data formats."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.synthetic_generator import (
    SyntheticGenerator, GenerationTask,
    SPS_CATEGORIES, COMPLEXITY_LEVELS,
)
from datasets.data_formats import (
    AlpacaExample, to_alpaca_format, to_sharegpt_format, alpaca_to_text,
    sharegpt_to_text, record_to_text, normalize_records_to_text,
)


# --- Data Formats ---

def test_alpaca_format():
    example = AlpacaExample(
        instruction="Write a PID controller in Structured Text",
        input="Setpoint: 100.0, PV range: 0-200",
        output="FUNCTION_BLOCK FB_PID\nVAR_INPUT\n  rSetpoint: REAL;\nEND_VAR",
    )
    formatted = to_alpaca_format(example)
    assert formatted["instruction"] == example.instruction
    assert "FB_PID" in formatted["output"]
    assert formatted["input"] == example.input


def test_sharegpt_format():
    result = to_sharegpt_format("Write PID", "Here is the code")
    assert len(result["conversations"]) == 2
    assert result["conversations"][0]["from"] == "human"
    assert result["conversations"][1]["from"] == "gpt"


def test_alpaca_to_text():
    example = AlpacaExample(
        instruction="Write code",
        input="context",
        output="result",
    )
    text = alpaca_to_text(example)
    assert "### Instruction:" in text
    assert "### Input:" in text
    assert "### Response:" in text


def test_alpaca_to_text_no_input():
    example = AlpacaExample(instruction="Write code", output="result")
    text = alpaca_to_text(example)
    assert "### Input:" not in text


def test_sharegpt_to_text():
    text = sharegpt_to_text(
        to_sharegpt_format("Write PID", "Here is the code")
    )
    assert "### User:" in text
    assert "### Assistant:" in text


def test_record_to_text_auto_prompt_completion():
    text = record_to_text({"prompt": "Write code", "completion": "done"})
    assert "### Prompt:" in text
    assert "### Completion:" in text


def test_normalize_records_to_text():
    rows = normalize_records_to_text([
        {"instruction": "Write code", "input": "", "output": "done"},
        {"text": "already normalized"},
    ])
    assert rows[0]["text"].startswith("### Instruction:")
    assert rows[1]["text"] == "already normalized"


# --- Generation Tasks ---

def test_generation_task():
    task = GenerationTask(
        category="pid_controller",
        complexity="basic",
        language="structured_text",
    )
    prompt = task.build_prompt()
    assert "basic" in prompt
    assert "pid controller" in prompt
    assert "structured text" in prompt


def test_sps_categories_not_empty():
    assert len(SPS_CATEGORIES) >= 20
    assert "pid_controller" in SPS_CATEGORIES
    assert "emergency_stop" in SPS_CATEGORIES


def test_complexity_levels():
    assert COMPLEXITY_LEVELS == ["basic", "intermediate", "advanced"]


# --- SyntheticGenerator ---

def test_synthetic_generator_init(tmp_path):
    from unittest.mock import MagicMock
    mock_provider = MagicMock()
    gen = SyntheticGenerator(provider=mock_provider, output_dir=tmp_path / "out")
    assert (tmp_path / "out").exists()


def test_synthetic_generator_batch(tmp_path):
    from unittest.mock import MagicMock
    mock_provider = MagicMock()
    mock_provider.chat.return_value = (
        "FUNCTION_BLOCK FB_PID\nVAR_INPUT\n  rSetpoint: REAL;\n"
        "END_VAR\nEND_FUNCTION_BLOCK"
    )
    gen = SyntheticGenerator(provider=mock_provider, output_dir=tmp_path / "out")
    examples = gen.generate_batch("pid_controller", "basic", count=2)
    assert len(examples) == 2
    assert "instruction" in examples[0]
    assert mock_provider.chat.call_count == 2
