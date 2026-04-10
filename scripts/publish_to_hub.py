#!/usr/bin/env python3
"""
Publish model to HuggingFace Hub with proper model card.

Usage:
    python scripts/publish_to_hub.py \
        --model-path ./outputs/my-model \
        --target-repo org/model-name \
        --template templates/model_card.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder

    HAS_HF = True
except ImportError:
    HAS_HF = False


def load_template(template_path: str) -> str:
    """Load and return model card template."""
    if not template_path or not Path(template_path).exists():
        # Default template
        return """---
language: en
license: apache-2.0
tags:
{{tags}}
---

# {{model_name}}

This model was fine-tuned with TuneForge.

## Model Details

- **Base Model:** {{base_model}}
- **Fine-tuning Method:** QLoRA
- **Status:** {{status}}

## Validation

{{validation_info}}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{{repo_id}}")
tokenizer = AutoTokenizer.from_pretrained("{{repo_id}}")
```

## Citation

If you use this model, please cite:

```bibtex
@software{tuneforge2026,
  title = {TuneForge: Benchmark-first Fine-tuning Framework},
  year = {2026},
  url = {https://github.com/AI-Engineerings/tuneforge}
}
```
"""

    return Path(template_path).read_text()


def render_model_card(
    template: str,
    repo_id: str,
    base_model: str,
    tags: list[str],
    validation_report_path: str | None,
) -> str:
    """Render model card with template variables."""
    # Parse validation report
    validation_info = "Validation report not provided."
    status = "🔧 Technical Preview (Unverified)"

    if validation_report_path and Path(validation_report_path).exists():
        try:
            report = json.loads(Path(validation_report_path).read_text())
            tier = report.get("tier", "unknown")
            run_count = len(report.get("verification_runs", []))

            if tier == "A" and run_count >= 2:
                status = "✅ Tier A Verified"
                validation_info = f"""
- **Tier:** A (Consumer/Pro Hardware)
- **Hardware:** {report.get("hardware_requirements", {}).get("min_gpu", "N/A")}
- **Runs:** {run_count} independent successful runs
- **Report:** See validation report in repository
"""
            else:
                status = f"⚠️ Tier {tier} (In Progress)"
                validation_info = f"Runs completed: {run_count}"
        except Exception as e:
            validation_info = f"Could not parse validation report: {e}"

    # Build tags YAML
    tags_yaml = "\n".join(f"- {tag}" for tag in tags)

    # Render template
    card = template.replace("{{model_name}}", repo_id.split("/")[-1])
    card = card.replace("{{repo_id}}", repo_id)
    card = card.replace("{{base_model}}", base_model)
    card = card.replace("{{tags}}", tags_yaml)
    card = card.replace("{{status}}", status)
    card = card.replace("{{validation_info}}", validation_info)
    card = card.replace("{{date}}", datetime.now().strftime("%Y-%m-%d"))

    return card


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--target-repo", required=True, help="Target HF Hub repository (org/name)")
    parser.add_argument("--template", help="Path to model card template")
    parser.add_argument("--tags", default="qlora,finetuned", help="Comma-separated tags")
    parser.add_argument("--validation-report", help="Path to validation report")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--base-model", default="unknown", help="Base model name")

    args = parser.parse_args()

    if not HAS_HF:
        print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
        return 1

    print(f"Publishing to: {args.target_repo}")
    print("=" * 60)

    # Load template and render model card
    template = load_template(args.template)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    model_card = render_model_card(
        template=template,
        repo_id=args.target_repo,
        base_model=args.base_model,
        tags=tags,
        validation_report_path=args.validation_report,
    )

    # Write model card to model directory
    model_path = Path(args.model_path)
    readme_path = model_path / "README.md"
    readme_path.write_text(model_card)
    print(f"✅ Generated model card: {readme_path}")

    # Copy validation report if provided
    if args.validation_report and Path(args.validation_report).exists():
        report_dest = model_path / "validation_report.json"
        import shutil

        shutil.copy(args.validation_report, report_dest)
        print(f"✅ Copied validation report: {report_dest}")

    # Create/upload to HF Hub
    _api = HfApi()

    try:
        create_repo(
            repo_id=args.target_repo,
            private=args.private,
            exist_ok=True,
        )
        print(f"✅ Repository ready: {args.target_repo}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return 1

    try:
        upload_folder(
            repo_id=args.target_repo,
            folder_path=str(model_path),
            commit_message=f"Upload model fine-tuned with TuneForge ({datetime.now().strftime('%Y-%m-%d')})",
        )
        print(f"✅ Model uploaded to: https://huggingface.co/{args.target_repo}")
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")
        return 1

    print("=" * 60)
    print("✅ Publishing complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
