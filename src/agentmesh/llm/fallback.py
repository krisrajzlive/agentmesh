"""Ordered failover across providers, guarded by per-provider circuit breakers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentmesh.llm.base import (
    ChatMessage,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)
from agentmesh.observability.logging import get_logger
from agentmesh.resilience import CircuitBreaker

log = get_logger(__name__)


class FallbackProvider:
    """Tries providers in priority order, skipping those whose circuit is open."""

    name = "fallback"

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
    ) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = list(providers)
        self._breakers = {
            p.name: CircuitBreaker(
                f"llm:{p.name}", failure_threshold=failure_threshold, reset_timeout=reset_timeout
            )
            for p in providers
        }

    @property
    def model(self) -> str:
        return self._providers[0].model

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        errors: list[str] = []
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            if not breaker.allow():
                errors.append(f"{provider.name}: circuit open")
                continue
            try:
                result = await provider.complete(
                    messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
                )
            except (LLMUnavailableError, LLMConfigurationError) as exc:
                breaker.record_failure()
                errors.append(str(exc))
                log.warning("llm_provider_failed", provider=provider.name, error=str(exc))
                continue
            except LLMError:
                # Request-specific failure (e.g. HTTP 400): another provider will not help.
                breaker.record_success()
                raise
            breaker.record_success()
            return result
        raise LLMUnavailableError(
            "all LLM providers failed: " + "; ".join(errors), provider=self.name
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        errors: list[str] = []
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            if not breaker.allow():
                errors.append(f"{provider.name}: circuit open")
                continue
            started = False
            try:
                async for chunk in provider.stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    started = True
                    yield chunk
            except (LLMUnavailableError, LLMConfigurationError) as exc:
                breaker.record_failure()
                if started:
                    raise  # cannot splice a different model into a half-delivered answer
                errors.append(str(exc))
                continue
            breaker.record_success()
            return
        raise LLMUnavailableError(
            "all LLM providers failed: " + "; ".join(errors), provider=self.name
        )

    async def aclose(self) -> None:
        for provider in self._providers:
            await provider.aclose()
