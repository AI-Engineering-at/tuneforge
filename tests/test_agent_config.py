"""Tests for AgentConfig and ExperimentBudget."""
import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_config import AgentConfig, ExperimentBudget


def test_budget_not_exhausted():
    budget = ExperimentBudget(max_experiments=10, max_hours=2.0)
    assert not budget.is_exhausted(5, 1.0)


def test_budget_exhausted_by_experiments():
    budget = ExperimentBudget(max_experiments=10, max_hours=2.0)
    assert budget.is_exhausted(10, 0.5)


def test_budget_exhausted_by_hours():
    budget = ExperimentBudget(max_experiments=100, max_hours=2.0)
    assert budget.is_exhausted(5, 2.0)


def test_budget_defaults():
    budget = ExperimentBudget()
    assert budget.max_experiments == 200
    assert budget.max_hours == 12.0


@patch("openai.OpenAI")
def test_config_defaults(mock_openai):
    config = AgentConfig()
    assert config.provider == "ollama"
    assert config.model == "qwen2.5-coder:7b"
    assert config.max_tokens == 4096
    assert config.time_budget_seconds == 300
    assert config.improvement_threshold == 0.001
    assert config.primary_metric == "val_bpb"
    assert config.metric_goal == "minimize"


@patch("openai.OpenAI")
def test_config_provider_default_model(mock_openai):
    config = AgentConfig(provider="ollama")
    assert config.model == "qwen2.5-coder:7b"


@patch("anthropic.Anthropic")
def test_config_claude_default_model(mock_anthropic):
    config = AgentConfig(provider="claude", api_key="test-key")
    assert config.model == "claude-sonnet-4-20250514"


@patch("openai.OpenAI")
def test_config_custom_model_preserved(mock_openai):
    config = AgentConfig(provider="ollama", model="llama3.1:8b")
    assert config.model == "llama3.1:8b"


@patch("openai.OpenAI")
def test_config_env_var_claude(mock_openai):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
        config = AgentConfig(provider="claude")
        assert config.api_key == "sk-test-123"


@patch("openai.OpenAI")
def test_config_env_var_openrouter(mock_openai):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-456"}):
        config = AgentConfig(provider="openrouter")
        assert config.api_key == "or-test-456"


@patch("openai.OpenAI")
def test_config_ollama_no_key_needed(mock_openai):
    config = AgentConfig(provider="ollama")
    assert config.api_key == "" or config.api_key is not None


@patch("openai.OpenAI")
def test_config_rejects_bad_metric_goal(mock_openai):
    with pytest.raises(ValueError):
        AgentConfig(metric_goal="sideways")
