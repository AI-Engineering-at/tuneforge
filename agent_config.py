"""Configuration for the autonomous Karpathy research loop.

Supports multiple LLM providers. Default: Ollama (free, local).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from providers import PROVIDER_REGISTRY


@dataclass
class ExperimentBudget:
    max_experiments: int = 200
    max_hours: float = 12.0

    def is_exhausted(self, experiments_run: int, hours_elapsed: float) -> bool:
        return experiments_run >= self.max_experiments or hours_elapsed >= self.max_hours


@dataclass
class AgentConfig:
    # LLM Provider
    provider: str = "ollama"  # claude | openai | openrouter | kimi | ollama
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    model: str = ""  # empty = use provider default
    base_url: str = ""  # empty = use provider default
    max_tokens: int = 4096

    # Experiment settings
    time_budget_seconds: int = 300  # 5 min per experiment
    improvement_threshold: float = 0.001  # min val_bpb improvement to keep
    primary_metric: str = "val_bpb"
    metric_goal: str = "minimize"
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)

    # Paths
    train_script: Path = Path("train.py")
    results_file: Path = Path("results/results.tsv")
    program_file: Path = Path("programs/program-conservative.md")
    work_dir: Path = Path(".")

    # Git
    branch_prefix: str = "autoresearch"
    auto_commit: bool = True

    # Logging
    log_file: Path = Path("results/agent_loop.log")
    verbose: bool = True

    def __post_init__(self):
        # Set provider-specific default model
        if not self.model:
            self.model = PROVIDER_REGISTRY.get(self.provider, {}).get("default_model", "qwen2.5-coder:7b")
        if self.metric_goal not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported metric_goal: {self.metric_goal}")
        # Provider-specific env var shortcuts
        if not self.api_key:
            env_map = {
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "kimi": "MOONSHOT_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "ollama": "",
            }
            env_var = env_map.get(self.provider, "LLM_API_KEY")
            if env_var:
                self.api_key = os.environ.get(env_var, "")
