"""Additional tests to achieve 100% coverage for data_formats.py."""
import pytest
from data_utils.data_formats import (
    record_to_text,
    normalize_records_to_text,
    SUPPORTED_DATASET_FORMATS,
)


def test_record_to_text_unsupported_format():
    """Test that unsupported format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported dataset_format"):
        record_to_text({}, dataset_format="unknown_format")


def test_record_to_text_sharegpt_format():
    """Test sharegpt format conversion."""
    record = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
    }
    result = record_to_text(record, dataset_format="sharegpt")
    assert "### User:" in result
    assert "Hello" in result
    assert "### Assistant:" in result
    assert "Hi there!" in result


def test_record_to_text_sharegpt_auto_detection():
    """Test auto-detection of sharegpt format."""
    record = {
        "conversations": [
            {"from": "user", "value": "Question?"},
            {"from": "assistant", "value": "Answer."},
        ]
    }
    result = record_to_text(record, dataset_format="auto")
    assert "### User:" in result
    assert "Question?" in result


def test_record_to_text_no_matching_format():
    """Test that record with unknown keys raises ValueError."""
    record = {"unknown_key": "value", "another_key": 123}
    with pytest.raises(ValueError, match="Could not normalize record"):
        record_to_text(record, dataset_format="auto")


def test_record_to_text_text_format_with_text_key():
    """Test text format with 'text' key."""
    record = {"text": "  Some content  "}
    result = record_to_text(record, dataset_format="text")
    assert result == "Some content"


def test_record_to_text_text_format_with_content_key():
    """Test text format with 'content' key."""
    record = {"content": "  Content here  "}
    result = record_to_text(record, dataset_format="text")
    assert result == "Content here"


def test_record_to_text_text_format_with_prompt_completion():
    """Test text format with prompt/completion keys."""
    record = {"prompt": "The prompt", "completion": "The completion"}
    result = record_to_text(record, dataset_format="text")
    assert "### Prompt:" in result
    assert "The prompt" in result
    assert "### Completion:" in result
    assert "The completion" in result


def test_record_to_text_text_format_empty_prompt_completion():
    """Test text format with empty prompt/completion raises ValueError."""
    record = {"prompt": "", "completion": ""}
    with pytest.raises(ValueError, match="Could not normalize record"):
        record_to_text(record, dataset_format="auto")


def test_record_to_text_auto_prefers_alpaca_over_sharegpt():
    """Test that auto format prefers alpaca when both keys present."""
    record = {
        "instruction": "Do something",
        "output": "Result",
        "conversations": [{"from": "human", "value": "Hello"}],
    }
    result = record_to_text(record, dataset_format="auto")
    assert "### Instruction:" in result
    assert "Do something" in result


def test_normalize_records_to_text_empty_list():
    """Test normalizing empty list returns empty list."""
    result = normalize_records_to_text([], dataset_format="alpaca")
    assert result == []


def test_normalize_records_to_text_multiple_records():
    """Test normalizing multiple records."""
    records = [
        {"instruction": "Task 1", "output": "Result 1"},
        {"instruction": "Task 2", "output": "Result 2"},
    ]
    result = normalize_records_to_text(records, dataset_format="alpaca")
    assert len(result) == 2
    assert all("text" in r for r in result)
    assert "Task 1" in result[0]["text"]
    assert "Task 2" in result[1]["text"]


def test_supported_dataset_formats():
    """Test that expected formats are supported."""
    assert "auto" in SUPPORTED_DATASET_FORMATS
    assert "alpaca" in SUPPORTED_DATASET_FORMATS
    assert "sharegpt" in SUPPORTED_DATASET_FORMATS
    assert "text" in SUPPORTED_DATASET_FORMATS
