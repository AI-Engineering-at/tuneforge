#!/usr/bin/env python3
"""
Generate Ollama Modelfile for a fine-tuned model.

Usage:
    python scripts/generate_modelfile.py --model-repo org/model-name --output modelfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def generate_modelfile(
    model_repo: str,
    base_model: str | None = None,
    parameters: dict[str, str] | None = None,
    template: str | None = None,
) -> str:
    """Generate Ollama Modelfile content."""
    # Default parameters
    params = {
        "temperature": "0.7",
        "top_p": "0.9",
        "top_k": "40",
        "num_ctx": "4096",
    }
    if parameters:
        params.update(parameters)

    # Default chat template
    chat_template = """{{- if .System }}
<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
    if template:
        chat_template = template

    # Build Modelfile
    lines = [
        "# TuneForge generated Modelfile",
        f"# Model: {model_repo}",
        "",
        f"FROM {base_model or 'llama3'}:8b",
        "",
        "# Adapter weights from HuggingFace Hub",
        f"ADAPTER https://huggingface.co/{model_repo}",
        "",
        "# System prompt",
        'SYSTEM """',
        "You are a helpful AI assistant fine-tuned with TuneForge.",
        '"""',
        "",
        "# Chat template",
        'TEMPLATE """',
        chat_template,
        '"""',
        "",
    ]

    # Add parameters
    for key, value in params.items():
        lines.append(f"PARAMETER {key} {value}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile")
    parser.add_argument("--model-repo", required=True, help="HF Hub repository (org/name)")
    parser.add_argument("--base-model", help="Base model tag (e.g., llama3:8b)")
    parser.add_argument("--output", default="modelfile", help="Output file name")
    parser.add_argument("--template-file", help="Path to custom chat template file")

    args = parser.parse_args()

    # Load custom template if provided
    chat_template = None
    if args.template_file and Path(args.template_file).exists():
        chat_template = Path(args.template_file).read_text()
        print(f"Loaded custom template from {args.template_file}")

    # Generate Modelfile
    modelfile = generate_modelfile(
        model_repo=args.model_repo,
        base_model=args.base_model,
        template=chat_template,
    )

    # Write output
    output_path = Path(args.output)
    output_path.write_text(modelfile)

    print(f"✅ Generated Modelfile: {output_path}")
    print("\n" + "=" * 60)
    print("To use with Ollama:")
    print(f"  ollama create my-model -f {output_path}")
    print("  ollama run my-model")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
