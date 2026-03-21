"""Tests for TuneForge public-repo documentation and audit tooling."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scripts.check_compliance_docs as compliance_mod
import scripts.check_docs_parity as parity_mod
import scripts.check_repo_hygiene as hygiene_mod
import scripts.check_template_completeness as templates_mod
import scripts.validate_audit_pack as audit_mod


def test_check_docs_parity_accepts_minimal_pair(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme_de = tmp_path / "README-DE.md"
    readme.write_text(
        "# TuneForge\n"
        "- Language: EN\n"
        "- Audience: Public\n"
        "- Last Sync: 2026-03-19\n"
        "- Pair: README-DE.md\n",
        encoding="utf-8",
    )
    readme_de.write_text(
        "# TuneForge\n"
        "- Language: DE\n"
        "- Audience: Public\n"
        "- Last Sync: 2026-03-19\n"
        "- Pair: README.md\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(parity_mod, "ROOT", tmp_path)
    monkeypatch.setattr(parity_mod, "DOC_PAIRS", {"README.md": "README-DE.md"})

    assert parity_mod.check_docs_parity() == []


def test_check_repo_hygiene_rejects_forbidden_artifacts():
    tracked = [
        ".env.example",
        "Dockerfile",
        "Dockerfile.finetune",
        "docker-compose.yml",
        "docker-compose.finetune.yml",
        ".env",
        "output/model.gguf",
        "results/protocol.jsonl",
    ]
    errors = hygiene_mod.check_repo_hygiene(tracked)
    assert any(".env" in error for error in errors)
    assert any("output/model.gguf" in error for error in errors)
    assert any("results/protocol.jsonl" in error for error in errors)


def test_check_template_completeness_accepts_minimal_templates(tmp_path, monkeypatch):
    en = tmp_path / "templates" / "MODEL_CARD_TEMPLATE-EN.md"
    de = tmp_path / "templates" / "MODEL_CARD_TEMPLATE-DE.md"
    en.parent.mkdir()
    en.write_text(
        "# Template\n"
        "## Required Fields\n"
        "- a\n"
        "## Completion Notes\n"
        "- b\n",
        encoding="utf-8",
    )
    de.write_text(
        "# Template\n"
        "## Pflichtfelder\n"
        "- a\n"
        "## Hinweise zur Ausfuellung\n"
        "- b\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(templates_mod, "ROOT", tmp_path)
    monkeypatch.setattr(templates_mod, "EN_TEMPLATES", {"templates/MODEL_CARD_TEMPLATE-EN.md"})
    monkeypatch.setattr(templates_mod, "DE_TEMPLATES", {"templates/MODEL_CARD_TEMPLATE-DE.md"})

    assert templates_mod.check_template_completeness() == []


def test_check_compliance_docs_accepts_required_tokens(tmp_path, monkeypatch):
    en = tmp_path / "COMPLIANCE_STATEMENT.md"
    de = tmp_path / "COMPLIANCE_STATEMENT-DE.md"
    en.write_text("Austria DSGVO EU AI Act not legal advice", encoding="utf-8")
    de.write_text("Oesterreich DSGVO EU AI Act keine Rechtsberatung", encoding="utf-8")

    monkeypatch.setattr(compliance_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        compliance_mod,
        "DOC_RULES",
        {
            "COMPLIANCE_STATEMENT.md": ["Austria", "DSGVO", "EU AI Act", "not legal advice"],
            "COMPLIANCE_STATEMENT-DE.md": ["Oesterreich", "DSGVO", "EU AI Act", "keine Rechtsberatung"],
        },
    )

    assert compliance_mod.check_compliance_docs() == []


def test_validate_audit_pack_accepts_minimal_examples(tmp_path, monkeypatch):
    audit_root = tmp_path / "audit"
    examples = audit_root / "examples"
    examples.mkdir(parents=True)

    (audit_root / "protocol.schema.json").write_text(
        json.dumps({"required": ["timestamp", "run_id", "event_type", "stage", "status", "message", "git_sha"]}),
        encoding="utf-8",
    )
    (audit_root / "siem-export.schema.json").write_text(
        json.dumps({"required": ["event_time", "event_source", "event_category", "severity", "message", "run_id", "git_sha"]}),
        encoding="utf-8",
    )
    (examples / "protocol.sample.jsonl").write_text(
        json.dumps({
            "timestamp": "2026-03-19T08:00:00Z",
            "run_id": "r1",
            "event_type": "run.started",
            "stage": "train",
            "status": "started",
            "message": "started",
            "git_sha": "abc1234",
        }) + "\n",
        encoding="utf-8",
    )
    (examples / "error-registry.sample.json").write_text(
        json.dumps([{
            "error_id": "ERR-1",
            "severity": "medium",
            "summary": "Summary",
            "mitigation": "Mitigation",
            "last_reviewed": "2026-03-19",
        }]),
        encoding="utf-8",
    )
    (examples / "incident-log.sample.json").write_text(
        json.dumps([{
            "incident_id": "INC-1",
            "severity": "high",
            "opened_at": "2026-03-19T09:00:00Z",
            "status": "closed",
            "summary": "Summary",
            "impact": "Impact",
        }]),
        encoding="utf-8",
    )
    (examples / "release-decision-log.sample.json").write_text(
        json.dumps([{
            "release_id": "REL-1",
            "decision": "publish",
            "status": "technical_preview",
            "approved_by": ["engineering-review"],
            "evidence_refs": ["validation/registry.json"],
        }]),
        encoding="utf-8",
    )
    (examples / "siem-event.sample.json").write_text(
        json.dumps({
            "event_time": "2026-03-19T09:15:00Z",
            "event_source": "tuneforge",
            "event_category": "validation",
            "severity": "info",
            "message": "Registry preview.",
            "run_id": "r1",
            "git_sha": "abc1234",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    monkeypatch.setattr(audit_mod, "AUDIT_ROOT", audit_root)
    monkeypatch.setattr(
        audit_mod,
        "REQUIRED_FILES",
        {
            audit_root / "protocol.schema.json",
            audit_root / "siem-export.schema.json",
            examples / "protocol.sample.jsonl",
            examples / "error-registry.sample.json",
            examples / "incident-log.sample.json",
            examples / "release-decision-log.sample.json",
            examples / "siem-event.sample.json",
        },
    )

    assert audit_mod.validate_audit_pack() == []
