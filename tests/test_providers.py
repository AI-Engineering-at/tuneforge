"""Tests for multi-LLM provider abstraction."""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers import (
    LLMProvider, AnthropicProvider, OpenAICompatProvider,
    create_provider, PROVIDER_REGISTRY
)


def test_provider_registry_has_all():
    for name in ["claude", "openai", "openrouter", "kimi", "ollama"]:
        assert name in PROVIDER_REGISTRY


def test_registry_has_default_models():
    assert PROVIDER_REGISTRY["claude"]["default_model"] == "claude-sonnet-4-20250514"
    assert PROVIDER_REGISTRY["openai"]["default_model"] == "gpt-4o"
    assert PROVIDER_REGISTRY["ollama"]["default_model"] == "qwen2.5-coder:7b"
    assert PROVIDER_REGISTRY["kimi"]["default_model"] == "kimi-k2.5"


def test_ollama_needs_no_key():
    assert PROVIDER_REGISTRY["ollama"]["needs_key"] is False


def test_cloud_providers_need_key():
    for name in ["claude", "openai", "openrouter", "kimi"]:
        assert PROVIDER_REGISTRY[name]["needs_key"] is True


@patch("openai.OpenAI")
def test_create_ollama_provider(mock_openai):
    provider = create_provider("ollama", model="qwen2.5-coder:7b")
    assert provider.name == "ollama"
    assert provider.model == "qwen2.5-coder:7b"
    assert "localhost:11434" in provider.base_url


@patch("openai.OpenAI")
def test_create_openrouter_provider(mock_openai):
    provider = create_provider(
        "openrouter", api_key="test-key",
        model="anthropic/claude-sonnet-4-20250514"
    )
    assert provider.name == "openrouter"
    assert "openrouter.ai" in provider.base_url
    assert "HTTP-Referer" in provider.extra_headers


@patch("openai.OpenAI")
def test_create_kimi_provider(mock_openai):
    provider = create_provider("kimi", api_key="test-key")
    assert provider.name == "kimi"
    assert "moonshot" in provider.base_url


@patch("anthropic.Anthropic")
def test_create_claude_provider(mock_anthropic):
    provider = create_provider("claude", api_key="test-key")
    assert provider.name == "claude"
    assert provider.model == "claude-sonnet-4-20250514"


@patch("openai.OpenAI")
def test_create_provider_with_custom_model(mock_openai):
    provider = create_provider("ollama", model="llama3.1:8b")
    assert provider.model == "llama3.1:8b"


@patch("openai.OpenAI")
def test_create_provider_with_custom_base_url(mock_openai):
    provider = create_provider(
        "ollama", base_url="http://localhost:11434/v1"
    )
    assert provider.base_url == "http://localhost:11434/v1"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nonexistent")


@patch("openai.OpenAI")
def test_openai_compat_default_model(mock_openai):
    provider = create_provider("openai", api_key="test")
    assert provider.model == "gpt-4o"


@patch("openai.OpenAI")
def test_ollama_no_api_key(mock_openai):
    provider = create_provider("ollama")
    assert provider.api_key == ""


@patch("openai.OpenAI")
def test_openai_compat_chat(mock_openai_cls):
    """Test that chat() calls the OpenAI client correctly."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="modified train.py code"))]
    )

    provider = create_provider("ollama", model="qwen2.5-coder:7b")
    result = provider.chat([{"role": "user", "content": "Optimize train.py"}])

    assert result == "modified train.py code"
    mock_client.chat.completions.create.assert_called_once()


@patch("anthropic.Anthropic")
def test_anthropic_chat_separates_system(mock_anthropic_cls):
    """Test that AnthropicProvider correctly separates system messages."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="optimized code")]
    )

    provider = create_provider("claude", api_key="test-key")
    result = provider.chat([
        {"role": "system", "content": "You are a researcher"},
        {"role": "user", "content": "Optimize train.py"},
    ])

    assert result == "optimized code"
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a researcher"
    # System message should NOT be in the messages array
    assert all(m["role"] != "system" for m in call_kwargs["messages"])
