#!/usr/bin/env python3
"""Drive a remote RTX 3090 validation run over Windows OpenSSH."""

from __future__ import annotations

import argparse
import base64
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone


DEFAULT_REMOTE_HOST = "joe@10.40.10.90"
DEFAULT_REMOTE_REPO_DIR = r"C:\Users\Joe\Documents\tuneforge"
DEFAULT_REPO_URL = "https://github.com/AI-Engineerings-at/tuneforge.git"
DEFAULT_BRANCH = "main"
DEFAULT_BASELINE_IMAGE = "ai-engineering/tuneforge-studio:technical-preview"
DEFAULT_BUNDLE_DIR = r"output\local-preview"
DEFAULT_PROTOCOL_FILE = r"results\protocol.jsonl"


@dataclass(frozen=True)
class RemoteConfig:
    remote_host: str = DEFAULT_REMOTE_HOST
    remote_repo_dir: str = DEFAULT_REMOTE_REPO_DIR
    repo_url: str = DEFAULT_REPO_URL
    branch: str = DEFAULT_BRANCH
    baseline_image: str = DEFAULT_BASELINE_IMAGE
    bundle_dir: str = DEFAULT_BUNDLE_DIR
    protocol_file: str = DEFAULT_PROTOCOL_FILE
    max_examples: int = 500
    tester_id: str = "pilot-3090-a"
    tester_organization: str = ""
    submission_kind: str = "local_smoke"
    result_status: str = "smoke"
    model_name: str = "local/tuneforge-legal-dsgvo-control"
    training_data: str = "OpenLegalData German court decisions"
    dataset_label: str = "OpenLegalData"
    docker_image_label: str = "local-windows-docker-compose"
    domain: str = "Legal / DSGVO"
    base_model: str = "LeoLM/leo-mistral-hessianai-7b"
    language: str = "de"
    output_name: str = "openlegaldata"
    independent_environment: bool = False
    run_id: str = ""


def _default_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return f"{prefix}-{timestamp}"


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def encode_powershell(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def run_remote_powershell(host: str, script: str) -> subprocess.CompletedProcess[str]:
    encoded = encode_powershell(script)
    return subprocess.run(
        ["ssh", host, "powershell", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_success(result: subprocess.CompletedProcess[str], context: str) -> None:
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"{context} failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def build_repo_bootstrap_script(config: RemoteConfig) -> str:
    repo_dir = powershell_quote(config.remote_repo_dir)
    repo_url = powershell_quote(config.repo_url)
    branch = powershell_quote(config.branch)
    return f"""
$ErrorActionPreference = 'Stop'
$repoDir = {repo_dir}
$repoUrl = {repo_url}
$branch = {branch}
if (Test-Path (Join-Path $repoDir '.git')) {{
    $status = (& git -C $repoDir status --porcelain).Trim()
    if ($status) {{
        throw "Remote repo is dirty at $repoDir. Refusing to update automatically."
    }}
    & git -C $repoDir fetch origin | Out-Host
    & git -C $repoDir checkout $branch | Out-Host
    & git -C $repoDir pull --ff-only origin $branch | Out-Host
}} else {{
    $parent = Split-Path -Parent $repoDir
    if (-not (Test-Path $parent)) {{
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }}
    & git clone --branch $branch $repoUrl $repoDir | Out-Host
}}
& git -C $repoDir rev-parse HEAD
"""


def build_preflight_script(config: RemoteConfig) -> str:
    repo_dir = powershell_quote(config.remote_repo_dir)
    run_id = powershell_quote(config.run_id or _default_run_id("preflight"))
    return f"""
$ErrorActionPreference = 'Stop'
$repoDir = {repo_dir}
$runId = {run_id}
if (-not (Test-Path $repoDir)) {{
    throw "Remote repo missing at $repoDir"
}}
Set-Location $repoDir
New-Item -ItemType Directory -Force -Path 'results' | Out-Null
$gpuLine = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Select-Object -First 1).Trim()
$gpuParts = $gpuLine -split ',\\s*'
$cudaLine = (& nvidia-smi | Select-String 'CUDA Version:' | Select-Object -First 1).ToString()
$cudaVersion = if ($cudaLine -match 'CUDA Version:\\s*([0-9.]+)') {{ $Matches[1] }} else {{ 'unknown' }}
$drive = Get-PSDrive -Name C
$facts = [ordered]@{{
    run_id = $runId
    captured_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    hostname = $env:COMPUTERNAME
    user = $env:USERNAME
    remote_repo_dir = $repoDir
    git_sha = (& git -C $repoDir rev-parse HEAD).Trim()
    gpu_model = $gpuParts[0]
    gpu_vram = $gpuParts[1]
    driver_version = $gpuParts[2]
    cuda_version = $cudaVersion
    docker_version = (& docker --version).Trim()
    git_version = (& git --version).Trim()
    python_version = (& py --version).Trim()
    free_disk_gb = [math]::Round($drive.Free / 1GB, 2)
}}
$factsPath = Join-Path $repoDir ('results\\' + $runId + '-preflight.json')
$facts | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $factsPath
$facts | ConvertTo-Json -Depth 4
"""


def build_legal_pilot_script(config: RemoteConfig) -> str:
    run_id = config.run_id or _default_run_id("pilot-3090")
    repo_dir = powershell_quote(config.remote_repo_dir)
    protocol_file = powershell_quote(config.protocol_file)
    bundle_dir = powershell_quote(config.bundle_dir)
    baseline_image = powershell_quote(config.baseline_image)
    output_name = powershell_quote(config.output_name)
    model_name = powershell_quote(config.model_name)
    training_data = powershell_quote(config.training_data)
    dataset_label = powershell_quote(config.dataset_label)
    docker_image_label = powershell_quote(config.docker_image_label)
    tester_id = powershell_quote(config.tester_id)
    tester_organization = powershell_quote(config.tester_organization)
    submission_kind = powershell_quote(config.submission_kind)
    result_status = powershell_quote(config.result_status)
    domain = powershell_quote(config.domain)
    base_model = powershell_quote(config.base_model)
    language = powershell_quote(config.language)
    run_id_q = powershell_quote(run_id)
    independent_flag = "$true" if config.independent_environment else "$false"

    return f"""
$ErrorActionPreference = 'Stop'
$repoDir = {repo_dir}
$protocolFile = {protocol_file}
$bundleDir = {bundle_dir}
$baselineImage = {baseline_image}
$runId = {run_id_q}
$outputName = {output_name}
$modelName = {model_name}
$trainingData = {training_data}
$datasetLabel = {dataset_label}
$dockerImageLabel = {docker_image_label}
$testerId = {tester_id}
$testerOrganization = {tester_organization}
$submissionKind = {submission_kind}
$resultStatus = {result_status}
$domain = {domain}
$baseModel = {base_model}
$language = {language}
$independentEnvironment = {independent_flag}

function Write-ProtocolEvent {{
    param(
        [string]$EventType,
        [string]$Stage,
        [string]$Status,
        [string]$Message,
        [string]$GitSha,
        [string]$ArtifactPath = '',
        [string]$MetricName = '',
        [double]$MetricValue = 0.0,
        [switch]$HasMetric
    )
    $args = @(
        (Join-Path $repoDir 'scripts\\write_protocol_event.py'),
        '--protocol-file', (Join-Path $repoDir $protocolFile),
        '--run-id', $runId,
        '--event-type', $EventType,
        '--stage', $Stage,
        '--status', $Status,
        '--message', $Message,
        '--git-sha', $GitSha,
        '--hardware-tier', 'tier_a_rtx_3090_24gb',
        '--public-status', 'technical_preview'
    )
    if ($ArtifactPath) {{
        $args += @('--artifact-path', $ArtifactPath)
    }}
    if ($HasMetric) {{
        $args += @('--metric-name', $MetricName, '--metric-value', [string]$MetricValue)
    }}
    & py -3 @args | Out-Host
}}

if (-not (Test-Path $repoDir)) {{
    throw "Remote repo missing at $repoDir"
}}
Set-Location $repoDir
foreach ($relative in @('results', 'output', 'datasets\\generated\\legal', $bundleDir)) {{
    New-Item -ItemType Directory -Force -Path (Join-Path $repoDir $relative) | Out-Null
}}

$gitSha = (& git -C $repoDir rev-parse HEAD).Trim()
$gpuLine = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Select-Object -First 1).Trim()
$gpuParts = $gpuLine -split ',\\s*'
$gpuModel = $gpuParts[0]
$gpuVramText = $gpuParts[1]
$driverVersion = $gpuParts[2]
$cudaLine = (& nvidia-smi | Select-String 'CUDA Version:' | Select-Object -First 1).ToString()
$cudaVersion = if ($cudaLine -match 'CUDA Version:\\s*([0-9.]+)') {{ $Matches[1] }} else {{ 'unknown' }}
$pythonVersion = ((& py --version).Trim() -replace '^Python\\s+', '')
$gpuVramGb = if ($gpuVramText -match '([0-9]+)') {{ [math]::Round([double]$Matches[1] / 1024, 1) }} else {{ 24.0 }}
$preflight = [ordered]@{{
    run_id = $runId
    captured_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    hostname = $env:COMPUTERNAME
    user = $env:USERNAME
    git_sha = $gitSha
    gpu_model = $gpuModel
    gpu_vram = $gpuVramText
    driver_version = $driverVersion
    cuda_version = $cudaVersion
    docker_version = (& docker --version).Trim()
    python_version = (& py --version).Trim()
}}
$preflightPath = Join-Path $repoDir ('results\\' + $runId + '-preflight.json')
$preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $preflightPath

Write-ProtocolEvent -EventType 'run.started' -Stage 'baseline' -Status 'started' -Message 'Remote RTX 3090 legacy baseline started.' -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-baseline.log')
try {{
    & docker build -f Dockerfile -t $baselineImage . | Tee-Object -FilePath (Join-Path $repoDir ('results\\' + $runId + '-baseline-build.log')) | Out-Host
    $baselineResultsDir = Join-Path $repoDir 'results'
    $baselineMount = $baselineResultsDir + ':/app/results'
    $baselineInner = "mkdir -p /app/results && cp /app/patches/train_consumer.py /app/train.py && cp /app/patches/prepare_consumer.py /app/prepare.py && AUTORESEARCH_TIER_CONFIG=/app/configs/tier3-24gb.json python3 train.py 2>&1 | tee /app/results/$runId-baseline.log"
    & docker run --rm --gpus all --entrypoint bash -v $baselineMount -v 'tuneforge-cache:/root/.cache/tuneforge' $baselineImage -lc $baselineInner | Out-Host
    Write-ProtocolEvent -EventType 'run.finished' -Stage 'baseline' -Status 'passed' -Message 'Legacy baseline finished on remote RTX 3090.' -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-baseline.log')
}} catch {{
    Write-ProtocolEvent -EventType 'run.failed' -Stage 'baseline' -Status 'failed' -Message $_.Exception.Message -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-baseline.log')
    throw
}}

$datasetPath = Join-Path $repoDir 'datasets\\generated\\legal'
& py -3 (Join-Path $repoDir 'datasets\\legal_data.py') download-openlegaldata --output-dir $datasetPath --max-examples {config.max_examples} --output-name $outputName | Tee-Object -FilePath (Join-Path $repoDir ('results\\' + $runId + '-dataset-download.json')) | Out-Host

Write-ProtocolEvent -EventType 'run.started' -Stage 'train' -Status 'started' -Message 'Legal DSGVO control fine-tune started.' -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-legal-train.log')
try {{
    $trainInner = "python3 -m finetune.trainer --config /app/finetune/configs/legal-dsgvo.yaml --eval --summary-json /app/results/$runId-training-summary.json 2>&1 | tee /app/results/$runId-legal-train.log"
    & docker compose -f docker-compose.finetune.yml run --rm --entrypoint bash finetune -lc $trainInner | Out-Host
    $summaryPath = Join-Path $repoDir ('results\\' + $runId + '-training-summary.json')
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-ProtocolEvent -EventType 'run.finished' -Stage 'train' -Status 'passed' -Message 'Legal DSGVO control fine-tune finished.' -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-legal-train.log') -MetricName $summary.primary_metric_name -MetricValue ([double]$summary.primary_metric_value) -HasMetric
}} catch {{
    Write-ProtocolEvent -EventType 'run.failed' -Stage 'train' -Status 'failed' -Message $_.Exception.Message -GitSha $gitSha -ArtifactPath ('results\\' + $runId + '-legal-train.log')
    throw
}}

Write-ProtocolEvent -EventType 'validation.started' -Stage 'bundle' -Status 'started' -Message 'Release bundle generation started.' -GitSha $gitSha -ArtifactPath $bundleDir
try {{
    $summaryPath = Join-Path $repoDir ('results\\' + $runId + '-training-summary.json')
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    $bundlePath = Join-Path $repoDir $bundleDir
    $bundleArgs = @(
        '-m', 'finetune.model_publisher', 'bundle',
        '--adapter-path', $bundlePath,
        '--model-name', $modelName,
        '--base-model', $baseModel,
        '--language', $language,
        '--domain', $domain,
        '--training-data', $trainingData,
        '--dataset', $datasetLabel,
        '--primary-metric', $summary.primary_metric_name,
        '--metric-goal', $summary.metric_goal,
        '--primary-value', [string]$summary.primary_metric_value,
        '--hardware', 'RTX 3090',
        '--hardware-tier', 'tier_a_rtx_3090_24gb',
        '--gpu-model', $gpuModel,
        '--gpu-vram-gb', [string]$gpuVramGb,
        '--driver-version', $driverVersion,
        '--cuda-version', $cudaVersion,
        '--docker-image', $dockerImageLabel,
        '--python-version', $pythonVersion,
        '--tester-id', $testerId,
        '--tester-organization', $testerOrganization,
        '--submission-kind', $submissionKind,
        '--result-status', $resultStatus,
        '--metrics-json', $summaryPath,
        '--peak-vram-mb', [string]$summary.peak_vram_mb,
        '--training-seconds', [string]$summary.training_seconds,
        '--git-sha', $gitSha,
        '--base-model-license', 'unknown',
        '--limitation', 'Human review required.'
    )
    if ($independentEnvironment) {{
        $bundleArgs += '--independent-environment'
    }}
    & py -3 @bundleArgs | Out-Host
    & py -3 (Join-Path $repoDir 'scripts\\validate_release_artifacts.py') $bundlePath | Out-Host
    Write-ProtocolEvent -EventType 'validation.accepted' -Stage 'bundle' -Status 'passed' -Message 'Release bundle validated successfully.' -GitSha $gitSha -ArtifactPath $bundleDir
}} catch {{
    Write-ProtocolEvent -EventType 'validation.rejected' -Stage 'bundle' -Status 'failed' -Message $_.Exception.Message -GitSha $gitSha -ArtifactPath $bundleDir
    throw
}}

Write-Output ('RUN_ID=' + $runId)
Write-Output ('BUNDLE_DIR=' + $bundleDir)
Write-Output ('PROTOCOL_FILE=' + $protocolFile)
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote RTX 3090 pilot runner for TuneForge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(common_parser: argparse.ArgumentParser) -> None:
        common_parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
        common_parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
        common_parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
        common_parser.add_argument("--branch", default=DEFAULT_BRANCH)

    bootstrap = subparsers.add_parser("bootstrap", help="Clone or fast-forward the remote repo")
    add_common(bootstrap)

    preflight = subparsers.add_parser("preflight", help="Collect remote 3090 facts into results/")
    add_common(preflight)
    preflight.add_argument("--run-id", default="")

    pilot = subparsers.add_parser("legal-pilot", help="Run the remote 3090 baseline + legal smoke flow")
    add_common(pilot)
    pilot.add_argument("--baseline-image", default=DEFAULT_BASELINE_IMAGE)
    pilot.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR)
    pilot.add_argument("--protocol-file", default=DEFAULT_PROTOCOL_FILE)
    pilot.add_argument("--max-examples", type=int, default=500)
    pilot.add_argument("--tester-id", default="pilot-3090-a")
    pilot.add_argument("--tester-organization", default="")
    pilot.add_argument("--submission-kind", default="local_smoke")
    pilot.add_argument("--result-status", default="smoke")
    pilot.add_argument("--model-name", default="local/tuneforge-legal-dsgvo-control")
    pilot.add_argument("--training-data", default="OpenLegalData German court decisions")
    pilot.add_argument("--dataset-label", default="OpenLegalData")
    pilot.add_argument("--docker-image-label", default="local-windows-docker-compose")
    pilot.add_argument("--domain", default="Legal / DSGVO")
    pilot.add_argument("--base-model", default="LeoLM/leo-mistral-hessianai-7b")
    pilot.add_argument("--language", default="de")
    pilot.add_argument("--output-name", default="openlegaldata")
    pilot.add_argument("--independent-environment", action="store_true")
    pilot.add_argument("--run-id", default="")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        remote_host=args.remote_host,
        remote_repo_dir=args.remote_repo_dir,
        repo_url=args.repo_url,
        branch=args.branch,
        baseline_image=getattr(args, "baseline_image", DEFAULT_BASELINE_IMAGE),
        bundle_dir=getattr(args, "bundle_dir", DEFAULT_BUNDLE_DIR),
        protocol_file=getattr(args, "protocol_file", DEFAULT_PROTOCOL_FILE),
        max_examples=getattr(args, "max_examples", 500),
        tester_id=getattr(args, "tester_id", "pilot-3090-a"),
        tester_organization=getattr(args, "tester_organization", ""),
        submission_kind=getattr(args, "submission_kind", "local_smoke"),
        result_status=getattr(args, "result_status", "smoke"),
        model_name=getattr(args, "model_name", "local/tuneforge-legal-dsgvo-control"),
        training_data=getattr(args, "training_data", "OpenLegalData German court decisions"),
        dataset_label=getattr(args, "dataset_label", "OpenLegalData"),
        docker_image_label=getattr(args, "docker_image_label", "local-windows-docker-compose"),
        domain=getattr(args, "domain", "Legal / DSGVO"),
        base_model=getattr(args, "base_model", "LeoLM/leo-mistral-hessianai-7b"),
        language=getattr(args, "language", "de"),
        output_name=getattr(args, "output_name", "openlegaldata"),
        independent_environment=getattr(args, "independent_environment", False),
        run_id=getattr(args, "run_id", ""),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)

    bootstrap_result = run_remote_powershell(config.remote_host, build_repo_bootstrap_script(config))
    ensure_success(bootstrap_result, "remote repo bootstrap")
    print(bootstrap_result.stdout.strip())

    if args.command == "bootstrap":
        return 0
    if args.command == "preflight":
        preflight_result = run_remote_powershell(config.remote_host, build_preflight_script(config))
        ensure_success(preflight_result, "remote preflight")
        print(preflight_result.stdout.strip())
        return 0

    pilot_result = run_remote_powershell(config.remote_host, build_legal_pilot_script(config))
    ensure_success(pilot_result, "remote legal pilot")
    print(pilot_result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
