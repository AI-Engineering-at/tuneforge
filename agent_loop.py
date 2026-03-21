"""Autonomous Karpathy research loop — provider-agnostic.

Works with ANY LLM provider (Claude, OpenAI, OpenRouter, Kimi, Ollama).
The agent reads program.md, proposes changes to train.py, runs experiments,
and keeps or discards based on val_bpb improvement.

Usage:
    # Free (local Ollama):
    python agent_loop.py --provider ollama --model qwen2.5-coder:7b

    # Cloud (Claude):
    ANTHROPIC_API_KEY=sk-... python agent_loop.py --provider claude

    # Cloud (OpenRouter — any model):
    OPENROUTER_API_KEY=sk-... python agent_loop.py --provider openrouter --model anthropic/claude-sonnet-4-20250514
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from agent_config import AgentConfig, ExperimentBudget
from providers import LLMProvider, create_provider

logger = logging.getLogger(__name__)

# --- Graceful Shutdown ---
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning(f"Received {sig_name} — finishing current experiment then stopping...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# --- Data Types ---

@dataclass
class ExperimentResult:
    val_bpb: float = 0.0
    primary_metric_name: str = "val_bpb"
    primary_metric_value: float = 0.0
    metric_goal: str = "minimize"
    metrics: dict[str, float] = field(default_factory=dict)
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_m: float = 0.0
    num_steps: int = 0
    num_params_m: float = 0.0
    depth: int = 0
    success: bool = True
    error_message: str = ""
    commit_hash: str = ""
    description: str = ""
    status: str = ""  # keep, discard, crash


@dataclass
class CodeChange:
    code_diff: str
    description: str
    reasoning: str = ""


# --- Results Parser ---

class ResultsParser:
    METRIC_PATTERN = re.compile(r"^(\w+):\s+(.+)$", re.MULTILINE)
    CRASH_PATTERNS = [
        r"Traceback \(most recent call last\)",
        r"RuntimeError:",
        r"CUDA out of memory",
        r"signal: killed",
    ]

    @staticmethod
    def parse_output(output: str) -> ExperimentResult:
        result = ExperimentResult()
        for pattern in ResultsParser.CRASH_PATTERNS:
            if re.search(pattern, output):
                result.success = False
                lines = output.strip().split("\n")
                result.error_message = lines[-1] if lines else "Unknown error"
                return result

        metadata: dict[str, str] = {}
        numeric_metrics: dict[str, float] = {}

        for match in ResultsParser.METRIC_PATTERN.finditer(output):
            key, value = match.group(1), match.group(2).strip()

            if key in {"primary_metric_name", "metric_goal"}:
                metadata[key] = value
                continue

            try:
                parsed = float(value)
            except ValueError:
                logger.warning(f"Could not parse metric {key}={value}")
                continue

            numeric_metrics[key] = parsed
            if key == "val_bpb":
                result.val_bpb = parsed
            elif key == "primary_metric_value":
                result.primary_metric_value = parsed
            elif key == "training_seconds":
                result.training_seconds = parsed
            elif key == "total_seconds":
                result.total_seconds = parsed
            elif key == "peak_vram_mb":
                result.peak_vram_mb = parsed
            elif key == "mfu_percent":
                result.mfu_percent = parsed
            elif key == "total_tokens_M":
                result.total_tokens_m = parsed
            elif key == "num_steps":
                result.num_steps = int(parsed)
            elif key == "num_params_M":
                result.num_params_m = parsed
            elif key == "depth":
                result.depth = int(parsed)

        result.metrics = {
            key: value for key, value in numeric_metrics.items()
            if key not in {"primary_metric_value"}
        }
        result.primary_metric_name = metadata.get("primary_metric_name", "val_bpb")
        result.metric_goal = metadata.get("metric_goal", "minimize")

        if result.primary_metric_name == "val_bpb" and result.val_bpb != 0.0:
            result.primary_metric_value = result.val_bpb
        elif result.primary_metric_value == 0.0 and result.primary_metric_name in result.metrics:
            result.primary_metric_value = result.metrics[result.primary_metric_name]

        if result.primary_metric_name == "val_bpb":
            result.val_bpb = result.primary_metric_value

        if result.primary_metric_value == 0.0:
            result.success = False
            result.error_message = (
                f"No {result.primary_metric_name} found in output"
                if result.primary_metric_name
                else "No primary metric found in output"
            )
        return result

    @staticmethod
    def parse_tsv(content: str) -> list[ExperimentResult]:
        results = []
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return results
        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 6 and header[1:3] == ["metric_name", "metric_value"]:
                metric_value = float(parts[2])
                metric_name = parts[1]
                result = ExperimentResult(
                    commit_hash=parts[0],
                    primary_metric_name=metric_name,
                    primary_metric_value=metric_value,
                    metric_goal="minimize",
                    peak_vram_mb=float(parts[3]) * 1024 if float(parts[3]) < 100 else float(parts[3]),
                    status=parts[4],
                    description=parts[5],
                    success=parts[4] != "crash",
                    metrics={metric_name: metric_value},
                )
                if metric_name == "val_bpb":
                    result.val_bpb = metric_value
                results.append(result)
            elif len(parts) >= 5:
                metric_value = float(parts[1])
                results.append(ExperimentResult(
                    commit_hash=parts[0],
                    val_bpb=metric_value,
                    primary_metric_name="val_bpb",
                    primary_metric_value=metric_value,
                    metric_goal="minimize",
                    metrics={"val_bpb": metric_value},
                    peak_vram_mb=float(parts[2]) * 1024 if float(parts[2]) < 100 else float(parts[2]),
                    status=parts[3],
                    description=parts[4],
                    success=parts[3] != "crash",
                ))
        return results

    @staticmethod
    def best_result(
        results: list[ExperimentResult],
        metric_name: str = "val_bpb",
        metric_goal: str = "minimize",
    ) -> ExperimentResult:
        kept = [
            r for r in results
            if r.status == "keep" and r.success and r.primary_metric_name == metric_name
        ]
        if not kept:
            return ExperimentResult()
        if metric_goal == "maximize":
            return max(kept, key=lambda r: r.primary_metric_value)
        return min(kept, key=lambda r: r.primary_metric_value)


# --- Research Agent (provider-agnostic) ---

class ResearchAgent:
    """Provider-agnostic research agent. Works with ANY LLM provider."""

    def __init__(self, provider: LLMProvider, max_tokens: int = 4096):
        self.provider = provider
        self.max_tokens = max_tokens

    def _build_prompt(
        self,
        program: str,
        current_code: str,
        history: list[ExperimentResult],
        primary_metric: str,
        metric_goal: str,
    ) -> str:
        history_text = "\n".join(
            f"  {r.description}: {r.primary_metric_name}={r.primary_metric_value:.6f} ({r.status})"
            for r in history[-20:]
        )
        best = ResultsParser.best_result(history, primary_metric, metric_goal)
        metric_verb = "maximize" if metric_goal == "maximize" else "minimize"
        return f"""## Research Program
{program}

## Current Best Result
{primary_metric}: {best.primary_metric_value:.6f} (from: {best.description})

## Experiment History (last 20)
{history_text}

## Current train.py
```python
{current_code}
```

## Your Task
Propose ONE change to train.py that could {metric_verb} {primary_metric}.
Return the COMPLETE modified train.py inside ```python``` code fences.
After the code, add a line: DESCRIPTION: <one-line description of what you changed>"""

    def propose_change(
        self,
        program: str,
        current_code: str,
        history: list[ExperimentResult],
        primary_metric: str = "val_bpb",
        metric_goal: str = "minimize",
    ) -> CodeChange:
        prompt = self._build_prompt(
            program,
            current_code,
            history,
            primary_metric,
            metric_goal,
        )
        response_text = self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return self._parse_response(response_text)

    def _parse_response(self, text: str) -> CodeChange:
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        code_diff = code_match.group(1).strip() if code_match else ""

        desc_match = re.search(r"DESCRIPTION:\s*(.+)", text)
        description = desc_match.group(1).strip() if desc_match else "No description"

        reasoning = text[:text.find("```python")] if "```python" in text else ""

        return CodeChange(code_diff=code_diff, description=description,
                          reasoning=reasoning)


# --- Experiment Runner ---

class ExperimentRunner:
    def __init__(self, work_dir: Path = Path("."), train_script: str = "train.py"):
        self.work_dir = Path(work_dir)
        self.train_script = train_script

    def write_code(self, code: str):
        path = self.work_dir / self.train_script
        # Backup before overwrite
        backup = path.with_suffix(".py.bak")
        if path.exists():
            backup.write_text(path.read_text())
        path.write_text(code)

    def read_code(self) -> str:
        path = self.work_dir / self.train_script
        if not path.exists():
            raise FileNotFoundError(f"Train script not found: {path}")
        return path.read_text()

    def restore_backup(self):
        backup = (self.work_dir / self.train_script).with_suffix(".py.bak")
        target = self.work_dir / self.train_script
        if backup.exists():
            target.write_text(backup.read_text())
            logger.info("Restored train.py from backup")

    def git_commit(self, message: str) -> str:
        subprocess.run(["git", "add", self.train_script],
                        cwd=self.work_dir, capture_output=True)
        result = subprocess.run(["git", "commit", "-m", message],
                        cwd=self.work_dir, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Git commit failed: {result.stderr.strip()}")
            return "no-commit"
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=self.work_dir, capture_output=True, text=True
        )
        return result.stdout.strip()

    def git_revert(self):
        subprocess.run(["git", "reset", "--hard", "HEAD~1"],
                        cwd=self.work_dir, capture_output=True)

    def check_gpu(self) -> dict:
        """Pre-flight VRAM check. Returns GPU info or raises if no GPU."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {"error": "nvidia-smi failed", "available": False}
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            info = {
                "name": parts[0],
                "vram_total_mb": int(parts[1]),
                "vram_free_mb": int(parts[2]),
                "temp_c": int(parts[3]),
                "available": True,
            }
            if info["vram_free_mb"] < 4000:
                logger.warning(f"Low VRAM: {info['vram_free_mb']}MB free — training may OOM")
            if info["temp_c"] > 85:
                logger.warning(f"GPU hot: {info['temp_c']}°C — consider cooling before training")
            return info
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"error": "nvidia-smi not found", "available": False}

    def run_training(self, timeout: int = 600) -> tuple[str, int]:
        try:
            result = subprocess.run(
                ["python3", self.train_script],
                cwd=self.work_dir,
                capture_output=True, text=True,
                timeout=timeout,
            )
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "TIMEOUT: Training exceeded time limit", 1
        except OSError as e:
            return f"OS ERROR: {e}", 1

    def append_results_tsv(self, commit: str, result: ExperimentResult):
        tsv_path = self.work_dir / "results" / "results.tsv"
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if not tsv_path.exists():
            tsv_path.write_text(
                "commit\tmetric_name\tmetric_value\tmemory_gb\tstatus\tdescription\n"
            )
        memory_gb = result.peak_vram_mb / 1024
        with open(tsv_path, "a") as f:
            f.write(
                f"{commit}\t{result.primary_metric_name}\t"
                f"{result.primary_metric_value:.6f}\t{memory_gb:.1f}\t"
                f"{result.status}\t{result.description}\n"
            )

    def write_experiment_json(self, experiment_num: int, result: ExperimentResult):
        """Write detailed JSON log per experiment for protocol tracking."""
        log_dir = self.work_dir / "results" / "experiments"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "experiment": experiment_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit": result.commit_hash,
            "description": result.description,
            "status": result.status,
            "success": result.success,
            "primary_metric_name": result.primary_metric_name,
            "primary_metric_value": result.primary_metric_value,
            "metric_goal": result.metric_goal,
            "val_bpb": result.val_bpb,
            "metrics": result.metrics,
            "peak_vram_mb": result.peak_vram_mb,
            "training_seconds": result.training_seconds,
            "total_seconds": result.total_seconds,
            "mfu_percent": result.mfu_percent,
            "num_steps": result.num_steps,
            "num_params_m": result.num_params_m,
            "error_message": result.error_message,
        }
        path = log_dir / f"exp-{experiment_num:04d}.json"
        path.write_text(json.dumps(record, indent=2))
        # Also append to JSONL protocol
        protocol = self.work_dir / "results" / "protocol.jsonl"
        with open(protocol, "a") as f:
            f.write(json.dumps(record) + "\n")


# --- Main Loop ---

class AutoResearchLoop:
    def __init__(self, config: AgentConfig, agent: ResearchAgent = None):
        self.config = config
        if agent:
            self.agent = agent
        else:
            provider = create_provider(
                config.provider,
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
            )
            self.agent = ResearchAgent(provider=provider,
                                       max_tokens=config.max_tokens)
        self.runner = ExperimentRunner(
            work_dir=config.work_dir,
            train_script=str(config.train_script),
        )
        self.experiments_run = 0
        self.start_time = None

    def _load_program(self) -> str:
        return self.config.program_file.read_text()

    def _load_history(self) -> list[ExperimentResult]:
        tsv_path = self.config.work_dir / self.config.results_file
        if not tsv_path.exists():
            return []
        return ResultsParser.parse_tsv(tsv_path.read_text())

    def run(self):
        global _shutdown_requested
        self.start_time = time.time()
        program = self._load_program()
        provider_name = self.agent.provider.name
        self.consecutive_failures = 0

        # Pre-flight GPU check
        gpu_info = self.runner.check_gpu()
        if gpu_info.get("available"):
            logger.info(f"GPU: {gpu_info['name']} | "
                        f"VRAM: {gpu_info['vram_free_mb']}MB free / {gpu_info['vram_total_mb']}MB | "
                        f"Temp: {gpu_info['temp_c']}°C")
        else:
            logger.warning(f"GPU check failed: {gpu_info.get('error', 'unknown')} — training may fail")

        logger.info(
            f"Starting AutoResearch Loop (provider={provider_name}, "
            f"max={self.config.budget.max_experiments} experiments, "
            f"{self.config.budget.max_hours}h)"
        )

        while not self._should_stop():
            self.experiments_run += 1
            logger.info(f"\n{'='*60}\n"
                        f"Experiment {self.experiments_run} | "
                        f"Elapsed: {(time.time() - self.start_time)/60:.1f}min | "
                        f"Failures: {self.consecutive_failures}\n{'='*60}")

            try:
                self._run_one_experiment(program)
                self.consecutive_failures = 0
            except ConnectionError as e:
                self.consecutive_failures += 1
                logger.error(f"LLM provider connection failed: {e}")
                if self.consecutive_failures >= 5:
                    logger.critical("5 consecutive LLM failures — stopping loop")
                    break
                wait = min(60, 10 * self.consecutive_failures)
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)
            except Exception as e:
                self.consecutive_failures += 1
                logger.error(f"Experiment {self.experiments_run} failed: {e}", exc_info=True)
                if self.consecutive_failures >= 10:
                    logger.critical("10 consecutive failures — stopping loop")
                    break

        elapsed = (time.time() - self.start_time) / 3600
        history = self._load_history()
        best = ResultsParser.best_result(
            history,
            self.config.primary_metric,
            self.config.metric_goal,
        )
        kept = [r for r in history if r.status == "keep"]
        discarded = [r for r in history if r.status == "discard"]
        crashed = [r for r in history if r.status == "crash"]

        summary = (
            f"\n{'='*60}\n"
            f"LOOP COMPLETE\n"
            f"{'='*60}\n"
            f"  Experiments:  {self.experiments_run}\n"
            f"  Duration:     {elapsed:.1f}h\n"
            f"  Kept:         {len(kept)}\n"
            f"  Discarded:    {len(discarded)}\n"
            f"  Crashed:      {len(crashed)}\n"
            f"  Best {self.config.primary_metric}: {best.primary_metric_value:.6f} ({best.description})\n"
            f"{'='*60}"
        )
        logger.info(summary)

        if _shutdown_requested:
            logger.info("Shutdown was requested via signal — exiting cleanly")

    def _should_stop(self) -> bool:
        if _shutdown_requested:
            return True
        elapsed_hours = (time.time() - self.start_time) / 3600
        return self.config.budget.is_exhausted(
            self.experiments_run, elapsed_hours
        )

    def _run_one_experiment(self, program: str):
        current_code = self.runner.read_code()
        history = self._load_history()
        best = ResultsParser.best_result(
            history,
            self.config.primary_metric,
            self.config.metric_goal,
        )

        # 1. Ask LLM for a change
        provider_name = self.agent.provider.name
        logger.info(f"Asking {provider_name} for next experiment...")
        change = self.agent.propose_change(
            program,
            current_code,
            history,
            primary_metric=self.config.primary_metric,
            metric_goal=self.config.metric_goal,
        )

        if not change.code_diff:
            logger.warning("LLM returned empty code — skipping experiment")
            return

        logger.info(f"Proposal: {change.description}")

        # 2. Write code + commit (with backup)
        self.runner.write_code(change.code_diff)

        # Syntax check before committing
        try:
            compile(change.code_diff, self.runner.train_script, "exec")
        except SyntaxError as e:
            logger.warning(f"Syntax error in proposed code: {e} — reverting")
            self.runner.restore_backup()
            result = ExperimentResult(
                success=False, status="crash",
                error_message=f"SyntaxError: {e}",
                description=change.description,
            )
            self.runner.append_results_tsv("syntax-err", result)
            self.runner.write_experiment_json(self.experiments_run, result)
            return

        commit_hash = self.runner.git_commit(
            f"experiment: {change.description}"
        )

        # 3. Run training
        timeout = self.config.time_budget_seconds + 120
        logger.info(f"Running training (timeout: {timeout}s)...")
        t0 = time.time()
        output, returncode = self.runner.run_training(timeout=timeout)
        wall_time = time.time() - t0

        # 4. Parse results
        result = ResultsParser.parse_output(output)
        result.commit_hash = commit_hash
        result.description = change.description
        if result.total_seconds == 0:
            result.total_seconds = wall_time
        if result.primary_metric_name == "val_bpb" and self.config.primary_metric != "val_bpb":
            if self.config.primary_metric in result.metrics:
                result.primary_metric_name = self.config.primary_metric
                result.primary_metric_value = result.metrics[self.config.primary_metric]
        result.metric_goal = self.config.metric_goal
        if result.primary_metric_name == "val_bpb":
            result.val_bpb = result.primary_metric_value
        if (
            result.success
            and result.primary_metric_name != self.config.primary_metric
        ):
            result.success = False
            result.error_message = (
                f"Expected metric `{self.config.primary_metric}`, "
                f"got `{result.primary_metric_name}`"
            )

        # 5. Keep or discard
        if not result.success:
            result.status = "crash"
            logger.warning(f"CRASH: {result.error_message}")
            self.runner.git_revert()
            self.runner.restore_backup()
        else:
            improved = False
            if best.primary_metric_value > 0:
                if self.config.metric_goal == "maximize":
                    improved = (
                        result.primary_metric_value >
                        best.primary_metric_value + self.config.improvement_threshold
                    )
                else:
                    improved = (
                        result.primary_metric_value <
                        best.primary_metric_value - self.config.improvement_threshold
                    )

            if best.primary_metric_value == 0:
                result.status = "keep"
                logger.info(
                    f"KEEP (baseline): {result.primary_metric_name}="
                    f"{result.primary_metric_value:.6f}"
                )
            elif improved:
                result.status = "keep"
                if self.config.metric_goal == "maximize":
                    improvement = result.primary_metric_value - best.primary_metric_value
                else:
                    improvement = best.primary_metric_value - result.primary_metric_value
                logger.info(
                    f"KEEP: {result.primary_metric_name}={result.primary_metric_value:.6f} "
                    f"(improved by {improvement:.6f})"
                )
            else:
                result.status = "discard"
                delta = result.primary_metric_value - best.primary_metric_value
                logger.info(
                    f"DISCARD: {result.primary_metric_name}={result.primary_metric_value:.6f} "
                    f"(best={best.primary_metric_value:.6f}, delta={delta:+.6f})"
                )
                self.runner.git_revert()

        # 6. Log to results.tsv + JSON protocol
        self.runner.append_results_tsv(commit_hash, result)
        self.runner.write_experiment_json(self.experiments_run, result)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AutoResearch Loop — Multi-Provider Autonomous Research"
    )
    parser.add_argument(
        "--provider", default="ollama",
        choices=["claude", "openai", "openrouter", "kimi", "ollama"],
        help="LLM provider (default: ollama = free/local)"
    )
    parser.add_argument(
        "--model", default="",
        help="Model name (empty=provider default)"
    )
    parser.add_argument(
        "--base-url", default="",
        help="Custom API base URL (e.g. http://localhost:11434/v1 for remote Ollama)"
    )
    parser.add_argument(
        "--program", default="programs/program-conservative.md"
    )
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--max-experiments", type=int, default=200)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Seconds per experiment (default: 300)")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--primary-metric", default="val_bpb")
    parser.add_argument(
        "--metric-goal",
        default="minimize",
        choices=["maximize", "minimize"],
    )
    args = parser.parse_args()

    # Setup dual logging: console + file
    log_dir = Path(args.work_dir) / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )

    logger.info(f"Log file: {log_file}")
    logger.info(f"Args: {vars(args)}")

    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        program_file=Path(args.program),
        work_dir=Path(args.work_dir),
        time_budget_seconds=args.time_budget,
        primary_metric=args.primary_metric,
        metric_goal=args.metric_goal,
        budget=ExperimentBudget(
            max_experiments=args.max_experiments,
            max_hours=args.max_hours,
        ),
        improvement_threshold=args.threshold,
    )
    loop = AutoResearchLoop(config=config)
    loop.run()


if __name__ == "__main__":
    main()
