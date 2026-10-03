from __future__ import annotations

import pytest

from agentmesh.config import Settings
from agentmesh.llm import build_llm
from agentmesh.resilience import CircuitBreaker, CircuitState, TokenBucket


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_circuit_breaker_lifecycle():
    clock = Clock()
    breaker = CircuitBreaker("t", failure_threshold=2, reset_timeout=10, clock=clock)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow()

    clock.now = 11
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow()
    assert not breaker.allow()  # single probe at a time
    breaker.record_failure()  # failed probe re-opens immediately
    assert breaker.state is CircuitState.OPEN
    clock.now = 25
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


async def test_token_bucket_refills_over_time():
    clock = Clock()
    bucket = TokenBucket(rate=1, capacity=2, clock=clock)
    assert await bucket.try_acquire()
    assert await bucket.try_acquire()
    assert not await bucket.try_acquire()
    assert await bucket.retry_after() == pytest.approx(1.0)
    clock.now = 1.5
    assert await bucket.try_acquire()


def test_settings_parse_csv_and_validate():
    s = Settings(_env_file=None, llm_providers="openai, ollama", api_keys="a:1,b:2")  # type: ignore[call-arg]
    assert s.llm_providers == ["openai", "ollama"]
    assert s.api_key_map() == {"1": "a", "2": "b"}
    with pytest.raises(ValueError, match="unknown LLM provider"):
        Settings(_env_file=None, llm_providers="bard")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings(_env_file=None, auth_mode="api_key")  # type: ignore[call-arg]


def test_provider_chain_skips_unconfigured_providers():
    none = Settings(_env_file=None, llm_providers="openai,huggingface")  # type: ignore[call-arg]
    assert build_llm(none) is None

    some = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_providers="openai,huggingface,ollama",
        hf_token="hf_test",
    )
    chain = build_llm(some)
    assert [p.name for p in chain.providers] == ["huggingface", "ollama"]


def test_secrets_are_not_leaked_in_repr():
    s = Settings(_env_file=None, openai_api_key="sk-very-secret")  # type: ignore[call-arg]
    assert "sk-very-secret" not in repr(s)


def test_blank_values_mean_unset():
    s = Settings(  # type: ignore[call-arg]
        _env_file=None, openai_api_key="", jwt_secret="  ", registry_url="", mcp_server_command=""
    )
    assert s.openai_api_key is None
    assert s.jwt_secret is None
    assert s.registry_url is None
    assert build_llm(Settings(_env_file=None, llm_providers="openai", openai_api_key="")) is None  # type: ignore[call-arg]


def test_the_shipped_env_example_is_loadable(monkeypatch):
    from pathlib import Path

    for key in list(__import__("os").environ):
        if key.startswith("AGENTMESH_") or key in {"OPENAI_API_KEY", "HF_TOKEN", "OLLAMA_API_KEY"}:
            monkeypatch.delenv(key)
    example = Path(__file__).parent.parent / ".env.example"
    s = Settings(_env_file=example)  # type: ignore[call-arg]
    assert s.environment == "development"
    assert s.llm_providers == ["openai", "huggingface", "ollama"]
    assert s.openai_api_key is None and s.api_keys == [] and s.registry_url is None
