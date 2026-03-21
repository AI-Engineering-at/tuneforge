"""Tests for legal data loader."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.legal_data import LegalDataLoader, LegalDataConfig


def test_legal_config_defaults():
    config = LegalDataConfig()
    assert "openlegaldata" in config.sources
    assert config.language == "de"
    assert config.max_examples == 10000
    assert config.train_split == 0.9


def test_format_qa_pair():
    loader = LegalDataLoader(LegalDataConfig())
    example = loader._format_qa_pair(
        context="DSGVO Art. 5: Personenbezogene Daten muessen...",
        question="Was sind die Grundsaetze der Datenverarbeitung?",
        answer="Rechtmaessigkeit, Verarbeitung nach Treu und Glauben, Transparenz...",
    )
    assert "instruction" in example
    assert example["instruction"] == "Was sind die Grundsaetze der Datenverarbeitung?"
    assert "DSGVO" in example["input"]


def test_format_case():
    loader = LegalDataLoader(LegalDataConfig())
    case = {
        "content": "Das Gericht entscheidet..." * 100,
        "abstract": "Zusammenfassung des Urteils",
    }
    example = loader._format_case(case)
    assert "Analysiere" in example["instruction"]
    assert len(example["input"]) <= 4000
    assert example["output"] == "Zusammenfassung des Urteils"


def test_save(tmp_path):
    config = LegalDataConfig(output_dir=tmp_path / "legal")
    loader = LegalDataLoader(config)
    examples = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]
    loader.save(examples, "test")
    saved = (tmp_path / "legal" / "test.jsonl").read_text()
    assert saved.count("\n") == 2
