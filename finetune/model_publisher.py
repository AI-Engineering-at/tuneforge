"""Release bundle helpers for TuneForge fine-tuned models.

This module generates a governed publication bundle for Hugging Face, GitHub
Releases, and Ollama-compatible distribution.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PUBLIC_STATUS_TECHNICAL_PREVIEW = "technical_preview"
PUBLIC_STATUS_VERIFIED_3090 = "verified_rtx_3090"
PUBLIC_STATUS_VERIFIED_48GB = "verified_48gb_plus"
VALID_PUBLIC_STATUSES = (
    PUBLIC_STATUS_TECHNICAL_PREVIEW,
    PUBLIC_STATUS_VERIFIED_3090,
    PUBLIC_STATUS_VERIFIED_48GB,
)

HARDWARE_TIER_3090 = "tier_a_rtx_3090_24gb"
HARDWARE_TIER_48GB = "tier_b_48gb_plus"
HARDWARE_TIER_UNASSIGNED = "unassigned"
VALID_HARDWARE_TIERS = (
    HARDWARE_TIER_3090,
    HARDWARE_TIER_48GB,
    HARDWARE_TIER_UNASSIGNED,
)

VALID_RESULT_STATUSES = ("passed", "failed", "partial", "smoke")
VALID_SUBMISSION_KINDS = ("local_smoke", "private_pilot", "release_candidate")

PUBLIC_STATUS_LABELS = {
    PUBLIC_STATUS_TECHNICAL_PREVIEW: "Technical Preview",
    PUBLIC_STATUS_VERIFIED_3090: "Verified on RTX 3090",
    PUBLIC_STATUS_VERIFIED_48GB: "Verified on 48 GB+",
}
HARDWARE_TIER_LABELS = {
    HARDWARE_TIER_3090: "Tier A - RTX 3090 / 24 GB",
    HARDWARE_TIER_48GB: "Tier B - 48 GB+",
    HARDWARE_TIER_UNASSIGNED: "Unassigned",
}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "artifact"


def _render_metric(value: float) -> str:
    if 0 <= value <= 1:
        return f"{value:.0%}"
    return f"{value:.6f}"


def _write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_public_status(status: str) -> str:
    return PUBLIC_STATUS_LABELS.get(status, status)


def render_hardware_tier(hardware_tier: str) -> str:
    return HARDWARE_TIER_LABELS.get(hardware_tier, hardware_tier)


def infer_hardware_tier(hardware: str) -> str:
    value = hardware.lower()
    if "3090" in value:
        return HARDWARE_TIER_3090
    if any(token in value for token in ("a100", "h100", "a6000", "rtx 6000 ada", "48 gb", "80 gb")):
        return HARDWARE_TIER_48GB
    return HARDWARE_TIER_UNASSIGNED


@dataclass
class TrainingManifest:
    base_model: str
    dataset: str
    primary_metric: str
    metric_goal: str
    hardware: str
    seed: int
    peak_vram_mb: float
    training_seconds: float
    git_sha: str
    license: str
    limitations: list[str] = field(default_factory=list)
    backend: str = "peft_trl"
    version: str = "0.2.0"
    public_status: str = PUBLIC_STATUS_TECHNICAL_PREVIEW
    hardware_tier: str = HARDWARE_TIER_UNASSIGNED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    name: str
    dataset: str
    hardware: str
    primary_metric_name: str
    primary_metric_value: float
    metric_goal: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    public_status: str = PUBLIC_STATUS_TECHNICAL_PREVIEW
    hardware_tier: str = HARDWARE_TIER_UNASSIGNED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Summary - {self.name}",
            "",
            f"- Dataset: {self.dataset}",
            f"- Hardware: {self.hardware}",
            f"- Public status: `{render_public_status(self.public_status)}`",
            f"- Hardware tier: `{render_hardware_tier(self.hardware_tier)}`",
            f"- Primary metric: `{self.primary_metric_name}` ({self.metric_goal})",
            f"- Primary value: `{self.primary_metric_value:.6f}`",
        ]
        if self.notes:
            lines.extend(["", self.notes])
        if self.metrics:
            lines.extend([
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for key, value in sorted(self.metrics.items()):
                lines.append(f"| {key} | {_render_metric(value)} |")
        return "\n".join(lines) + "\n"


@dataclass
class LicenseManifest:
    release_license: str
    base_model_license: str
    upstream_components: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentManifest:
    hardware_tier: str
    hardware: str
    gpu_model: str
    gpu_vram_gb: float
    driver_version: str
    cuda_version: str
    os_name: str
    docker_image: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesterAttestation:
    __test__ = False

    tester_id: str
    tester_organization: str
    submission_kind: str
    result_status: str
    independent_environment: bool
    submitted_at: str
    notes: str = ""
    artifacts_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCard:
    model_name: str
    base_model: str
    language: str
    domain: str
    metrics: dict = field(default_factory=dict)
    training_data: str = ""
    license: str = "MIT"
    hardware: str = "NVIDIA RTX 3090 (24GB)"
    backend: str = "transformers + peft + trl"
    intended_use: str = "Domain-specific local inference and benchmarked adaptation."
    out_of_scope_use: str = "High-risk or legally sensitive automation without human review."
    limitations: list[str] = field(default_factory=lambda: [
        "Benchmark claims are valid only for the documented hardware budget.",
        "This release does not constitute legal advice or compliance certification.",
    ])
    safety_notes: list[str] = field(default_factory=lambda: [
        "Review outputs before production use.",
    ])
    privacy_notes: list[str] = field(default_factory=lambda: [
        "Do not assume the base model is safe for personal data without a separate review.",
    ])
    public_status: str = PUBLIC_STATUS_TECHNICAL_PREVIEW
    hardware_tier: str = HARDWARE_TIER_UNASSIGNED

    def to_markdown(self) -> str:
        metrics_table = "\n".join(
            f"| {k} | {_render_metric(float(v))} |"
            for k, v in sorted(self.metrics.items())
        ) or "| n/a | n/a |"
        limitations = "\n".join(f"- {item}" for item in self.limitations)
        safety_notes = "\n".join(f"- {item}" for item in self.safety_notes)
        privacy_notes = "\n".join(f"- {item}" for item in self.privacy_notes)
        status_note = (
            "This release is a Technical Preview until TuneForge completes the public hardware "
            "validation matrix."
            if self.public_status == PUBLIC_STATUS_TECHNICAL_PREVIEW else
            "This release is allowed to carry a verified hardware label only when the validation "
            "registry records enough independent successful runs for the same tier."
        )
        return f"""---
license: {self.license}
language: {self.language}
base_model: {self.base_model}
tags:
  - tuneforge
  - fine-tuned
  - qlora
  - {self.public_status.replace("_", "-")}
  - {self.domain.lower().replace(" ", "-").replace("/", "-")}
---

# {self.model_name}

TuneForge release bundle for **{self.domain}**.

> Public status: **{render_public_status(self.public_status)}**
>
> Validation tier: **{render_hardware_tier(self.hardware_tier)}**
>
> {status_note}

## Metrics

| Metric | Value |
|--------|-------|
{metrics_table}

## Training

- **Base Model:** {self.base_model}
- **Method:** QLoRA
- **Backend:** {self.backend}
- **Data:** {self.training_data}
- **Hardware:** {self.hardware}
- **Public Status:** {render_public_status(self.public_status)}
- **Validation Tier:** {render_hardware_tier(self.hardware_tier)}

## Intended Use

{self.intended_use}

## Out-of-Scope Use

{self.out_of_scope_use}

## Limitations

{limitations}

## Safety Notes

{safety_notes}

## Privacy Notes

{privacy_notes}

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{self.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.model_name}")
```

For Ollama-compatible distribution, see the generated `Modelfile`.
"""


def build_hf_repo_id(org: str, domain: str, base_model: str) -> str:
    model_part = base_model.split("/")[-1]
    return f"{org}/tuneforge-{_slugify(domain)}-{_slugify(model_part)}-adapter"


def build_ollama_model_name(domain: str, base_model: str, quantization: str) -> str:
    model_part = base_model.split("/")[-1]
    return f"tuneforge:{_slugify(domain)}-{_slugify(model_part)}-{_slugify(quantization)}"


def write_modelfile(
    output_path: str | Path,
    gguf_filename: str,
    ollama_model_name: str,
    system_prompt: str = "",
) -> Path:
    output_path = Path(output_path)
    lines = [f"FROM ./{gguf_filename}"]
    if system_prompt:
        escaped = system_prompt.replace('"', '\\"')
        lines.append(f'PARAMETER system "{escaped}"')
    lines.extend([
        f"# Suggested Ollama model name: {ollama_model_name}",
        "# Generate with: ollama create <name> -f Modelfile",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_release_bundle(
    adapter_path: str | Path,
    card: ModelCard,
    manifest: TrainingManifest,
    benchmark: BenchmarkSummary,
    license_manifest: LicenseManifest,
    environment_manifest: EnvironmentManifest,
    tester_attestation: TesterAttestation,
    gguf_filename: str = "",
    ollama_model_name: str = "",
    system_prompt: str = "",
) -> dict[str, Path]:
    adapter_dir = Path(adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    readme_path = adapter_dir / "README.md"
    manifest_path = adapter_dir / "training-manifest.json"
    benchmark_json_path = adapter_dir / "benchmark-summary.json"
    benchmark_md_path = adapter_dir / "benchmark-summary.md"
    license_manifest_path = adapter_dir / "license-manifest.json"
    environment_manifest_path = adapter_dir / "environment-manifest.json"
    tester_attestation_path = adapter_dir / "tester-attestation.json"

    readme_path.write_text(card.to_markdown(), encoding="utf-8")
    _write_json(manifest_path, manifest.to_dict())
    _write_json(benchmark_json_path, benchmark.to_dict())
    benchmark_md_path.write_text(benchmark.to_markdown(), encoding="utf-8")
    _write_json(license_manifest_path, license_manifest.to_dict())
    _write_json(environment_manifest_path, environment_manifest.to_dict())
    _write_json(tester_attestation_path, tester_attestation.to_dict())

    outputs = {
        "readme": readme_path,
        "training_manifest": manifest_path,
        "benchmark_json": benchmark_json_path,
        "benchmark_markdown": benchmark_md_path,
        "license_manifest": license_manifest_path,
        "environment_manifest": environment_manifest_path,
        "tester_attestation": tester_attestation_path,
    }

    if gguf_filename:
        modelfile_path = write_modelfile(
            adapter_dir / "Modelfile",
            gguf_filename=gguf_filename,
            ollama_model_name=ollama_model_name,
            system_prompt=system_prompt,
        )
        outputs["modelfile"] = modelfile_path

    return outputs


class HFPublisher:
    """Publish LoRA adapters to Hugging Face Hub."""

    def __init__(self, token: str = ""):
        self.token = token

    def publish(
        self,
        adapter_path: str,
        card: ModelCard,
        manifest: TrainingManifest,
        benchmark: BenchmarkSummary,
        license_manifest: LicenseManifest,
        environment_manifest: EnvironmentManifest,
        tester_attestation: TesterAttestation,
        gguf_filename: str = "",
        ollama_model_name: str = "",
        system_prompt: str = "",
    ):
        from finetune.zeroth_core import pre_publish_zeroth_check
        from huggingface_hub import HfApi

        # --- Zeroth Seam: Prevent unauthorized publishing ---
        from dataclasses import asdict
        pre_publish_zeroth_check(asdict(card), asdict(manifest))
        # ----------------------------------------------------

        write_release_bundle(
            adapter_path=adapter_path,
            card=card,
            manifest=manifest,
            benchmark=benchmark,
            license_manifest=license_manifest,
            environment_manifest=environment_manifest,
            tester_attestation=tester_attestation,
            gguf_filename=gguf_filename,
            ollama_model_name=ollama_model_name,
            system_prompt=system_prompt,
        )

        api = HfApi(token=self.token)
        api.create_repo(card.model_name, exist_ok=True)
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=card.model_name,
            commit_message=f"Publish {card.model_name} TuneForge release bundle",
        )
        logger.info("Published to https://huggingface.co/%s", card.model_name)


class GGUFConverter:
    """Convert release artifacts to GGUF format for Ollama-compatible use."""

    @staticmethod
    def convert(
        adapter_path: str,
        output_path: str,
        quantization: str = "q4_k_m",
        converter_command: str = "",
    ) -> str:
        command = converter_command or os.environ.get("GGUF_CONVERTER", "")
        if command:
            cmd = command.split() + [
                "--outtype", quantization,
                "--outfile", output_path,
                adapter_path,
            ]
        else:
            cmd = [
                "python3", "-m", "llama_cpp.convert",
                "--outtype", quantization,
                "--outfile", output_path,
                adapter_path,
            ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
        logger.info("GGUF saved to %s", output_path)
        return output_path


def _add_release_args(parser: argparse.ArgumentParser):
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--metric-goal", required=True, choices=["maximize", "minimize"])
    parser.add_argument("--primary-value", required=True, type=float)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-vram-mb", type=float, default=0.0)
    parser.add_argument("--training-seconds", type=float, default=0.0)
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--license", default="MIT")
    parser.add_argument("--base-model-license", default="unknown")
    parser.add_argument("--backend", default="peft_trl")
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--public-status", default=PUBLIC_STATUS_TECHNICAL_PREVIEW, choices=VALID_PUBLIC_STATUSES)
    parser.add_argument("--hardware-tier", default="")
    parser.add_argument("--gpu-model", default="")
    parser.add_argument("--gpu-vram-gb", type=float, default=0.0)
    parser.add_argument("--driver-version", default="unknown")
    parser.add_argument("--cuda-version", default="unknown")
    parser.add_argument("--os-name", default=platform.platform())
    parser.add_argument("--docker-image", default="unknown")
    parser.add_argument("--python-version", default=sys.version.split()[0])
    parser.add_argument("--tester-id", default="local-smoke")
    parser.add_argument("--tester-organization", default="")
    parser.add_argument("--submission-kind", default="local_smoke", choices=VALID_SUBMISSION_KINDS)
    parser.add_argument("--result-status", default="smoke", choices=VALID_RESULT_STATUSES)
    parser.add_argument("--submitted-at", default="")
    parser.add_argument("--tester-notes", default="")
    parser.add_argument("--independent-environment", action="store_true")
    parser.add_argument("--metrics-json")
    parser.add_argument("--notes", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--gguf-filename", default="")
    parser.add_argument("--ollama-model-name", default="")
    parser.add_argument("--limitation", action="append", default=[])


def _load_metrics(path: str | None, primary_metric: str, primary_value: float) -> dict[str, float]:
    metrics: dict[str, float] = {primary_metric: primary_value}
    if not path:
        return metrics

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        source = payload["metrics"]
    else:
        source = payload

    for key, value in source.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def build_release_objects(
    args: argparse.Namespace,
) -> tuple[
    ModelCard,
    TrainingManifest,
    BenchmarkSummary,
    LicenseManifest,
    EnvironmentManifest,
    TesterAttestation,
]:
    metrics = _load_metrics(args.metrics_json, args.primary_metric, args.primary_value)
    hardware_tier = args.hardware_tier or infer_hardware_tier(args.hardware)
    card = ModelCard(
        model_name=args.model_name,
        base_model=args.base_model,
        language=args.language,
        domain=args.domain,
        metrics=metrics,
        training_data=args.training_data,
        license=args.license,
        hardware=args.hardware,
        backend=args.backend,
        limitations=args.limitation or [
            "Benchmark claims apply only to the documented hardware budget.",
        ],
        public_status=args.public_status,
        hardware_tier=hardware_tier,
    )
    manifest = TrainingManifest(
        base_model=args.base_model,
        dataset=args.dataset,
        primary_metric=args.primary_metric,
        metric_goal=args.metric_goal,
        hardware=args.hardware,
        seed=args.seed,
        peak_vram_mb=args.peak_vram_mb,
        training_seconds=args.training_seconds,
        git_sha=args.git_sha,
        license=args.license,
        limitations=card.limitations,
        backend=args.backend,
        version=args.version,
        public_status=args.public_status,
        hardware_tier=hardware_tier,
    )
    benchmark = BenchmarkSummary(
        name=args.domain,
        dataset=args.dataset,
        hardware=args.hardware,
        primary_metric_name=args.primary_metric,
        primary_metric_value=args.primary_value,
        metric_goal=args.metric_goal,
        metrics=metrics,
        notes=args.notes,
        public_status=args.public_status,
        hardware_tier=hardware_tier,
    )
    license_manifest = LicenseManifest(
        release_license=args.license,
        base_model_license=args.base_model_license,
        upstream_components={
            "karpathy/autoresearch": "MIT",
            "transformers": "Apache-2.0",
            "trl": "Apache-2.0",
            "peft": "Apache-2.0",
            "unsloth": "Apache-2.0",
        },
    )
    environment_manifest = EnvironmentManifest(
        hardware_tier=hardware_tier,
        hardware=args.hardware,
        gpu_model=args.gpu_model or args.hardware,
        gpu_vram_gb=args.gpu_vram_gb,
        driver_version=args.driver_version,
        cuda_version=args.cuda_version,
        os_name=args.os_name,
        docker_image=args.docker_image,
        python_version=args.python_version,
    )
    tester_attestation = TesterAttestation(
        tester_id=args.tester_id,
        tester_organization=args.tester_organization,
        submission_kind=args.submission_kind,
        result_status=args.result_status,
        independent_environment=args.independent_environment,
        submitted_at=args.submitted_at or _current_timestamp(),
        notes=args.tester_notes,
    )
    return card, manifest, benchmark, license_manifest, environment_manifest, tester_attestation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TuneForge release bundle helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle", help="Write release bundle files")
    _add_release_args(bundle)

    publish = subparsers.add_parser("publish", help="Write bundle and publish to Hugging Face")
    _add_release_args(publish)
    publish.add_argument("--hf-token", default="")

    convert = subparsers.add_parser("convert", help="Convert an adapter to GGUF and write a Modelfile")
    convert.add_argument("--adapter-path", required=True)
    convert.add_argument("--output-path", required=True)
    convert.add_argument("--quantization", default="q4_k_m")
    convert.add_argument("--converter-command", default="")
    convert.add_argument("--ollama-model-name", default="")
    convert.add_argument("--system-prompt", default="")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if args.command == "convert":
        GGUFConverter.convert(
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            quantization=args.quantization,
            converter_command=args.converter_command,
        )
        if args.ollama_model_name:
            output_path = Path(args.output_path)
            write_modelfile(
                output_path=output_path.parent / "Modelfile",
                gguf_filename=output_path.name,
                ollama_model_name=args.ollama_model_name,
                system_prompt=args.system_prompt,
            )
        return 0

    card, manifest, benchmark, license_manifest, environment_manifest, tester_attestation = build_release_objects(args)
    ollama_model_name = args.ollama_model_name or (
        build_ollama_model_name(args.domain, args.base_model, "q4_k_m")
        if args.gguf_filename else ""
    )

    if args.command == "bundle":
        write_release_bundle(
            adapter_path=args.adapter_path,
            card=card,
            manifest=manifest,
            benchmark=benchmark,
            license_manifest=license_manifest,
            environment_manifest=environment_manifest,
            tester_attestation=tester_attestation,
            gguf_filename=args.gguf_filename,
            ollama_model_name=ollama_model_name,
            system_prompt=args.system_prompt,
        )
        return 0

    publisher = HFPublisher(token=args.hf_token or os.environ.get("HF_TOKEN", ""))
    publisher.publish(
        adapter_path=args.adapter_path,
        card=card,
        manifest=manifest,
        benchmark=benchmark,
        license_manifest=license_manifest,
        environment_manifest=environment_manifest,
        tester_attestation=tester_attestation,
        gguf_filename=args.gguf_filename,
        ollama_model_name=ollama_model_name,
        system_prompt=args.system_prompt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
