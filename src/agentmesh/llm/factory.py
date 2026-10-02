"""Build the configured provider chain from :class:`Settings`."""

from __future__ import annotations

from agentmesh.config import Settings
from agentmesh.llm.base import LLMProvider
from agentmesh.llm.fallback import FallbackProvider
from agentmesh.llm.ollama import OllamaProvider
from agentmesh.llm.openai_compat import OpenAICompatibleProvider
from agentmesh.observability.logging import get_logger

log = get_logger(__name__)


def build_llm(settings: Settings) -> LLMProvider | None:
    """Return a failover provider, or ``None`` when no usable provider is configured.

    Providers missing credentials are skipped with a warning rather than failing
    start-up, so a partially configured environment still serves traffic.
    """
    providers: list[LLMProvider] = []
    timeout = settings.llm_timeout_seconds
    retries = settings.llm_max_retries
    for name in settings.llm_providers:
        if name == "openai":
            if settings.openai_api_key is None:
                log.warning("llm_provider_skipped", provider=name, reason="OPENAI_API_KEY not set")
                continue
            providers.append(
                OpenAICompatibleProvider(
                    name="openai",
                    model=settings.openai_model,
                    base_url=settings.openai_base_url,
                    api_key=settings.openai_api_key.get_secret_value(),
                    timeout=timeout,
                    max_retries=retries,
                )
            )
        elif name == "huggingface":
            if settings.hf_token is None:
                log.warning("llm_provider_skipped", provider=name, reason="HF_TOKEN not set")
                continue
            providers.append(
                OpenAICompatibleProvider(
                    name="huggingface",
                    model=settings.hf_model,
                    base_url=settings.hf_base_url,
                    api_key=settings.hf_token.get_secret_value(),
                    timeout=timeout,
                    max_retries=retries,
                )
            )
        elif name == "ollama":
            key = settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else None
            providers.append(
                OllamaProvider(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                    api_key=key,
                    timeout=timeout,
                    max_retries=retries,
                )
            )
    if not providers:
        return None
    log.info("llm_chain_ready", providers=[p.name for p in providers])
    return FallbackProvider(providers)
