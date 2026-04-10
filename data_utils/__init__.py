"""Dataset processing modules for TuneForge.

This package provides data format converters and generators for fine-tuning.
Note: This is a local package, not to be confused with Hugging Face's datasets library.
"""

from data_utils.data_formats import (
    SUPPORTED_DATASET_FORMATS,
    AlpacaExample,
    alpaca_to_text,
    normalize_records_to_text,
    record_to_text,
    sharegpt_to_text,
    to_alpaca_format,
    to_sharegpt_format,
)

__all__ = [
    "SUPPORTED_DATASET_FORMATS",
    "AlpacaExample",
    "alpaca_to_text",
    "normalize_records_to_text",
    "record_to_text",
    "sharegpt_to_text",
    "to_alpaca_format",
    "to_sharegpt_format",
]
