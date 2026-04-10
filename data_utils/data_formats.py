"""Training data format converters.

Supports Alpaca, ShareGPT, and plain text records for fine-tuning datasets.
"""

from dataclasses import dataclass


SUPPORTED_DATASET_FORMATS = {"auto", "alpaca", "sharegpt", "text"}


@dataclass
class AlpacaExample:
    instruction: str
    input: str = ""
    output: str = ""


def to_alpaca_format(example: AlpacaExample) -> dict:
    return {
        "instruction": example.instruction,
        "input": example.input,
        "output": example.output,
    }


def to_sharegpt_format(instruction: str, response: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]
    }


def alpaca_to_text(example: AlpacaExample) -> str:
    """Convert Alpaca example to plain text for SFT training."""
    parts = [f"### Instruction:\n{example.instruction}"]
    if example.input:
        parts.append(f"\n### Input:\n{example.input}")
    parts.append(f"\n### Response:\n{example.output}")
    return "\n".join(parts)


def sharegpt_to_text(example: dict) -> str:
    """Convert a ShareGPT-style conversation into plain text for SFT training."""
    messages = []
    role_map = {
        "human": "User",
        "user": "User",
        "gpt": "Assistant",
        "assistant": "Assistant",
        "system": "System",
    }

    for turn in example.get("conversations", []):
        role = role_map.get(turn.get("from", "").lower(), "Message")
        value = turn.get("value", "").strip()
        if value:
            messages.append(f"### {role}:\n{value}")

    return "\n\n".join(messages)


def record_to_text(record: dict, dataset_format: str = "auto") -> str:
    """Normalize one record to the plain-text format consumed by SFT trainers."""
    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(f"Unsupported dataset_format: {dataset_format}")

    if dataset_format in {"auto", "alpaca"} and "instruction" in record:
        example = AlpacaExample(
            instruction=record.get("instruction", ""),
            input=record.get("input", ""),
            output=record.get("output", ""),
        )
        return alpaca_to_text(example)

    if dataset_format in {"auto", "sharegpt"} and "conversations" in record:
        return sharegpt_to_text(record)

    if dataset_format in {"auto", "text"}:
        for key in ("text", "content"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        prompt = record.get("prompt", "")
        completion = record.get("completion", "")
        if prompt or completion:
            return f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}".strip()

    raise ValueError(f"Could not normalize record with keys: {sorted(record.keys())}")


def normalize_records_to_text(records: list[dict], dataset_format: str = "auto") -> list[dict]:
    """Normalize a list of records into HuggingFace-ready `text` examples."""
    return [{"text": record_to_text(record, dataset_format=dataset_format)} for record in records]
