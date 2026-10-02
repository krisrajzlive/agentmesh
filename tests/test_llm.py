from __future__ import annotations

import httpx
import pytest
import respx

from agentmesh.llm import (
    ChatMessage,
    FallbackProvider,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitedError,
    LLMResponse,
    LLMUnavailableError,
    OllamaProvider,
    OpenAICompatibleProvider,
)

MESSAGES = [ChatMessage("user", "hi")]
OPENAI = "https://openai.test/v1"
OLLAMA = "http://ollama.test"


def _openai(retries: int = 1) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="openai", model="gpt-x", base_url=OPENAI, api_key="sk-test", max_retries=retries
    )


def _ollama() -> OllamaProvider:
    return OllamaProvider(model="llama", base_url=OLLAMA, max_retries=0)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", instant)


@respx.mock
async def test_openai_complete_parses_content_and_usage():
    route = respx.post(f"{OPENAI}/chat/completions").respond(
        json={
            "model": "gpt-x",
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )
    result = await _openai().complete(MESSAGES, json_mode=True, max_tokens=10)
    assert result.content == "hello"
    assert result.usage.total_tokens == 5
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-test"
    assert '"response_format":{"type":"json_object"}' in request.read().decode().replace(" ", "")


@respx.mock
async def test_openai_stream_yields_deltas():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{OPENAI}/chat/completions").respond(text=sse)
    assert [c async for c in _openai().stream(MESSAGES)] == ["Hel", "lo"]


@respx.mock
async def test_ollama_complete_and_stream():
    respx.post(f"{OLLAMA}/api/chat").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"message": {"content": "yo"}, "prompt_eval_count": 4, "eval_count": 1},
            ),
            httpx.Response(
                200,
                text='{"message":{"content":"a"}}\n{"message":{"content":"b"},"done":true}\n',
            ),
        ]
    )
    provider = _ollama()
    result = await provider.complete(MESSAGES)
    assert result.content == "yo"
    assert result.usage.prompt_tokens == 4
    assert [c async for c in provider.stream(MESSAGES)] == ["a", "b"]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, LLMConfigurationError),
        (429, LLMRateLimitedError),
        (503, LLMUnavailableError),
        (400, LLMError),
    ],
)
@respx.mock
async def test_http_errors_are_mapped(status, error):
    respx.post(f"{OPENAI}/chat/completions").respond(status)
    with pytest.raises(error):
        await _openai(retries=0).complete(MESSAGES)


@respx.mock
async def test_transient_errors_are_retried():
    route = respx.post(f"{OPENAI}/chat/completions").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    assert (await _openai().complete(MESSAGES)).content == "ok"
    assert route.call_count == 2


@respx.mock
async def test_rate_limits_are_not_retried_in_place():
    route = respx.post(f"{OPENAI}/chat/completions").respond(429)
    with pytest.raises(LLMRateLimitedError):
        await _openai(retries=3).complete(MESSAGES)
    assert route.call_count == 1


@respx.mock
async def test_connection_errors_become_unavailable():
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LLMUnavailableError):
        await _ollama().complete(MESSAGES)


class Scripted:
    def __init__(self, name: str, outcome: str | Exception) -> None:
        self.name = name
        self.model = name
        self._outcome = outcome
        self.calls = 0

    async def complete(self, messages, **_):
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return LLMResponse(self._outcome, self.name, self.name)

    async def stream(self, messages, **_):
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        yield self._outcome

    async def aclose(self):
        return None


async def test_fallback_uses_next_provider_on_failure():
    first = Scripted("a", LLMRateLimitedError("429", provider="a"))
    chain = FallbackProvider([first, Scripted("b", "from-b")])
    assert (await chain.complete(MESSAGES)).content == "from-b"
    assert [c async for c in chain.stream(MESSAGES)] == ["from-b"]


async def test_fallback_opens_circuit_after_repeated_failures():
    broken = Scripted("a", LLMUnavailableError("down", provider="a"))
    healthy = Scripted("b", "ok")
    chain = FallbackProvider([broken, healthy], failure_threshold=2, reset_timeout=3600)
    for _ in range(4):
        await chain.complete(MESSAGES)
    assert broken.calls == 2  # circuit opened; later calls skip it entirely
    assert healthy.calls == 4


async def test_fallback_does_not_mask_request_errors():
    other = Scripted("b", "ok")
    with pytest.raises(LLMError):
        await FallbackProvider([Scripted("a", LLMError("400", provider="a")), other]).complete(
            MESSAGES
        )
    assert other.calls == 0


async def test_fallback_reports_all_failures():
    chain = FallbackProvider(
        [
            Scripted("a", LLMUnavailableError("x", provider="a")),
            Scripted("b", LLMConfigurationError("y", provider="b")),
        ]
    )
    with pytest.raises(LLMUnavailableError, match="all LLM providers failed"):
        await chain.complete(MESSAGES)
