"""Shared HTTP plumbing for LLM providers: retries, error mapping, metrics."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentmesh.llm.base import (
    ChatMessage,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitedError,
    LLMResponse,
    LLMUnavailableError,
)
from agentmesh.observability.metrics import LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS


def raise_for_status(response: httpx.Response, provider: str) -> None:
    """Translate an HTTP error response into a typed :class:`LLMError`."""
    status = response.status_code
    if status < 400:
        return
    detail = response.text[:300]
    if status in (401, 403):
        raise LLMConfigurationError(
            f"{provider}: credentials rejected ({status})", provider=provider
        )
    if status == 429:
        raise LLMRateLimitedError(f"{provider}: rate limited", provider=provider)
    if status in (408, 409, 425) or status >= 500:
        raise LLMUnavailableError(
            f"{provider}: upstream error {status}: {detail}", provider=provider
        )
    raise LLMError(f"{provider}: request rejected ({status}): {detail}", provider=provider)


def _retryable(exc: BaseException) -> bool:
    # Rate limits are not retried in place: failing over to another provider is faster.
    return isinstance(exc, LLMUnavailableError) and not isinstance(exc, LLMRateLimitedError)


class HttpLLMProvider(ABC):
    """Base class for providers that speak HTTP/JSON."""

    name: str

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._owns_client = client is None
        # Auth is attached per request so an injected client (tests, shared pools) works too.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @abstractmethod
    async def _complete_once(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse: ...

    @abstractmethod
    def _stream_once(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int | None
    ) -> AsyncIterator[str]: ...

    async def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        try:
            response = await self._client.post(
                f"{self._base_url}{path}", json=payload, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(f"{self.name}: timeout", provider=self.name) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(
                f"{self.name}: connection error: {exc}", provider=self.name
            ) from exc
        raise_for_status(response, self.name)
        return response

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_retryable),
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential_jitter(initial=0.5, max=5.0),
                reraise=True,
            ):
                with attempt:
                    result = await self._complete_once(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
        except LLMError:
            LLM_REQUESTS.labels(self.name, "error").inc()
            raise
        LLM_REQUESTS.labels(self.name, "ok").inc()
        LLM_LATENCY.labels(self.name).observe(time.perf_counter() - started)
        LLM_TOKENS.labels(self.name, "prompt").inc(result.usage.prompt_tokens)
        LLM_TOKENS.labels(self.name, "completion").inc(result.usage.completion_tokens)
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            async for chunk in self._stream_once(
                messages, temperature=temperature, max_tokens=max_tokens
            ):
                yield chunk
        except LLMError:
            LLM_REQUESTS.labels(self.name, "error").inc()
            raise
        LLM_REQUESTS.labels(self.name, "ok").inc()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
