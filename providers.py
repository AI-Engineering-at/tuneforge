"""Multi-LLM provider abstraction for the AutoResearch loop.

Architecture: Only 2 actual implementations:
  AnthropicProvider    — for Claude (different message format, anthropic SDK)
  OpenAICompatProvider — for OpenAI/OpenRouter/Kimi/Ollama (openai SDK, different base_url)

Usage:
    provider = create_provider("ollama", model="qwen2.5-coder:7b")
    response = provider.chat([{"role": "user", "content": "Optimize train.py"}])

    provider = create_provider("claude", api_key=os.environ["ANTHROPIC_API_KEY"])
    response = provider.chat([{"role": "user", "content": "..."}])
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds — exponential: 2, 4, 8


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, messages: list[dict], max_tokens: int = 4096,
             temperature: float = 0.7) -> str:
        """Send messages, get response text. Universal interface."""
        ...


@dataclass
class AnthropicProvider(LLMProvider):
    """Claude via Anthropic API (different SDK, different message format)."""
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    _client: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "claude"

    def __post_init__(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, messages: list[dict], max_tokens: int = 4096,
             temperature: float = 0.7) -> str:
        # Anthropic: system message is separate, not in messages array
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"Claude API error (attempt {attempt}/{MAX_RETRIES}): {e} — retrying in {wait}s")
                    time.sleep(wait)
        raise ConnectionError(f"Claude API failed after {MAX_RETRIES} attempts: {last_error}")


@dataclass
class OpenAICompatProvider(LLMProvider):
    """Works with OpenAI, OpenRouter, Kimi, Ollama — all OpenAI-compatible.

    The openai Python SDK supports custom base_url, so we reuse it for
    ALL OpenAI-compatible providers. Only the base_url and headers differ.
    """
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    provider_name: str = "openai"
    extra_headers: dict = field(default_factory=dict)
    _client: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.provider_name

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key or "ollama",  # Ollama doesn't need a key
            base_url=self.base_url,
            default_headers=self.extra_headers or None,
        )

    def chat(self, messages: list[dict], max_tokens: int = 4096,
             temperature: float = 0.7) -> str:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"{self.provider_name} API error (attempt {attempt}/{MAX_RETRIES}): {e} — retrying in {wait}s")
                    time.sleep(wait)
        raise ConnectionError(f"{self.provider_name} API failed after {MAX_RETRIES} attempts: {last_error}")


# --- Provider Registry ---

PROVIDER_REGISTRY = {
    "claude": {
        "class": AnthropicProvider,
        "default_model": "claude-sonnet-4-20250514",
        "needs_key": True,
    },
    "openai": {
        "class": OpenAICompatProvider,
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "needs_key": True,
    },
    "openrouter": {
        "class": OpenAICompatProvider,
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "needs_key": True,
        "extra_headers": {
            "HTTP-Referer": "https://ai-engineering.at",
            "X-Title": "AutoResearch Kit",
        },
    },
    "kimi": {
        "class": OpenAICompatProvider,
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.5",
        "needs_key": True,
    },
    "gemini": {
        "class": OpenAICompatProvider,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "needs_key": True,
    },
    "ollama": {
        "class": OpenAICompatProvider,
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5-coder:7b",
        "needs_key": False,
    },
}


def create_provider(provider_name: str, api_key: str = "",
                    model: str = None, base_url: str = None) -> LLMProvider:
    """Factory: create a provider by name.

    Usage:
        provider = create_provider("ollama", model="qwen2.5-coder:7b")
        provider = create_provider("claude", api_key=os.environ["ANTHROPIC_API_KEY"])
        provider = create_provider("openrouter", api_key="sk-...", model="anthropic/claude-sonnet-4-20250514")
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(PROVIDER_REGISTRY.keys())}"
        )

    reg = PROVIDER_REGISTRY[provider_name]
    model = model or reg["default_model"]

    if provider_name == "claude":
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        return OpenAICompatProvider(
            api_key=api_key,
            model=model,
            base_url=base_url or reg.get("base_url", ""),
            provider_name=provider_name,
            extra_headers=reg.get("extra_headers", {}),
        )


# --- Fallback Provider ---

ENV_VAR_MAP = {
    "claude": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "kimi": ["MOONSHOT_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"],
    "ollama": [],
}

QUOTA_INDICATORS = [
    "429", "rate limit", "quota", "resource_exhausted",
    "RESOURCE_EXHAUSTED", "too many requests", "billing", "insufficient_quota",
]


def _is_quota_error(error):
    err_str = str(error).lower()
    return any(ind.lower() in err_str for ind in QUOTA_INDICATORS)


class FallbackProvider(LLMProvider):
    """Meta-provider: tries providers in order, auto-switches on failure."""

    def __init__(self, providers, labels=None):
        if not providers:
            raise ValueError("FallbackProvider needs at least 1 provider")
        self.providers = providers
        self.labels = labels or [f"{p.name}-{i}" for i, p in enumerate(providers)]
        self._active_index = 0
        self._exhausted = set()
        self._call_count = 0
        self._fallback_count = 0

    @property
    def name(self):
        active = self.providers[self._active_index]
        return f"fallback({active.name}|{self.labels[self._active_index]})"

    @property
    def active_provider(self):
        return self.providers[self._active_index]

    @property
    def stats(self):
        return {
            "total_calls": self._call_count,
            "fallbacks": self._fallback_count,
            "active_index": self._active_index,
            "active_label": self.labels[self._active_index],
            "exhausted": [self.labels[i] for i in self._exhausted],
        }

    def chat(self, messages, max_tokens=4096, temperature=0.7):
        self._call_count += 1
        errors = []
        for attempt in range(len(self.providers)):
            idx = (self._active_index + attempt) % len(self.providers)
            if idx in self._exhausted:
                continue
            provider = self.providers[idx]
            label = self.labels[idx]
            try:
                logger.info(f"Fallback chain: trying {label} (index {idx})")
                result = provider.chat(messages, max_tokens=max_tokens, temperature=temperature)
                if idx != self._active_index:
                    self._fallback_count += 1
                    logger.warning(f"Fallback activated: {self.labels[self._active_index]} -> {label}")
                    self._active_index = idx
                return result
            except ConnectionError as e:
                logger.warning(f"Provider {label} failed: {e}")
                if _is_quota_error(e):
                    logger.warning(f"Provider {label} QUOTA EXHAUSTED")
                    self._exhausted.add(idx)
                errors.append(f"{label}: {e}")
            except Exception as e:
                logger.warning(f"Provider {label} unexpected error: {e}")
                errors.append(f"{label}: {e}")
                if _is_quota_error(e):
                    self._exhausted.add(idx)
        error_log = "\n".join(f"  [{i+1}] {err}" for i, err in enumerate(errors))
        raise ConnectionError(f"ALL {len(self.providers)} providers failed:\n{error_log}")


def create_fallback_chain(provider_names=None, models=None):
    """Build a fallback chain. Default: Gemini key1 -> Gemini key2 -> Kimi."""
    if provider_names is None:
        provider_names = ["gemini", "gemini", "kimi"]
    if models is None:
        models = [""] * len(provider_names)
    elif len(models) < len(provider_names):
        models = models + [""] * (len(provider_names) - len(models))

    providers = []
    labels = []
    key_usage_count = {}

    for i, pname in enumerate(provider_names):
        if pname not in PROVIDER_REGISTRY:
            logger.warning(f"Unknown provider \'{pname}\' in fallback chain")
            continue
        env_vars = ENV_VAR_MAP.get(pname, [])
        count = key_usage_count.get(pname, 0)
        key_usage_count[pname] = count + 1
        api_key = ""
        key_label = ""
        if env_vars:
            if count < len(env_vars):
                env_var = env_vars[count]
                api_key = os.environ.get(env_var, "")
                key_label = env_var
            else:
                env_var = env_vars[-1]
                api_key = os.environ.get(env_var, "")
                key_label = f"{env_var}(reused)"
        reg = PROVIDER_REGISTRY[pname]
        if reg.get("needs_key") and not api_key:
            logger.warning(f"Fallback chain: skipping {pname} (no key in {key_label or 'env'})")
            continue
        model = models[i] or reg["default_model"]
        provider = create_provider(pname, api_key=api_key, model=model)
        label = f"{pname}-{key_label}" if key_label else pname
        providers.append(provider)
        labels.append(label)
        logger.info(f"Fallback chain [{len(providers)}]: {label} (model={model})")

    if not providers:
        raise ValueError("No providers available for fallback chain. Set API keys.")
    if len(providers) == 1:
        logger.warning("Fallback chain has only 1 provider -- no fallback available")
    return FallbackProvider(providers=providers, labels=labels)
