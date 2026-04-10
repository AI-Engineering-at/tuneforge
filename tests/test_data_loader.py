"""Tests for legal data loader."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_utils.legal_data import LegalDataLoader, LegalDataConfig, main


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


def test_download_openlegaldata_uses_fetch_json(monkeypatch, tmp_path):
    config = LegalDataConfig(output_dir=tmp_path / "legal", max_examples=2)
    loader = LegalDataLoader(config)
    pages = [
        {
            "results": [
                {"content": "Urteil A", "abstract": "Kurz A"},
                {"content": "Urteil B", "abstract": "Kurz B"},
            ],
            "next": "https://example.test/page/2",
        }
    ]

    def fake_fetch(url, params=None):
        return pages.pop(0)

    monkeypatch.setattr(loader, "_fetch_json", fake_fetch)
    examples = loader.download_openlegaldata()
    assert len(examples) == 2
    assert examples[0]["output"] == "Kurz A"


def test_cli_download_openlegaldata(monkeypatch, tmp_path, capsys):
    fake_examples = [
        {"instruction": "Q1", "input": "I1", "output": "A1"},
        {"instruction": "Q2", "input": "I2", "output": "A2"},
    ]

    monkeypatch.setattr(LegalDataLoader, "download_openlegaldata", lambda self: fake_examples)

    rc = main(
        [
            "download-openlegaldata",
            "--output-dir",
            str(tmp_path / "legal"),
            "--max-examples",
            "2",
            "--output-name",
            "cases",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["example_count"] == 2
    assert (tmp_path / "legal" / "cases.jsonl").exists()
