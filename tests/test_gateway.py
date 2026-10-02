from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal

import httpx
import pytest

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.client import AgentClient
from agentmesh.config import Settings
from agentmesh.gateway import (
    InMemoryRateLimiter,
    RedisFixedWindowLimiter,
    create_gateway_app,
)
from agentmesh.orchestrator import StaticDirectory
from agentmesh.runtime import create_agent_app
from tests.support import Switchboard

FX_URL = "http://gw.test/agents/fx-conversion-agent"
UPSTREAM_KEY = "svc-upstream-key"


def settings_with(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


@asynccontextmanager
async def gateway(*, limit: int = 100, down: tuple[str, ...] = ()):
    """Gateway in front of two key-protected agents. Clients hold their own keys, not the agents'."""
    agent_settings = settings_with(auth_mode="api_key", api_keys=f"gateway:{UPSTREAM_KEY}")
    fx = create_agent_app(
        FX_SPEC,
        build_fx_executor(agent_settings, rates=StaticRates({"EUR": Decimal("0.9")})),
        agent_settings,
        public_url="http://fx.test",
    )
    batch = create_agent_app(
        BATCH_SPEC, BatchExecutor(), agent_settings, public_url="http://batch.test"
    )
    gw_settings = settings_with(
        auth_mode="api_key",
        api_keys="alice:alice-key,bob:bob-key",
        outbound_api_key=UPSTREAM_KEY,
        gateway_rate_limit_per_minute=limit,
        gateway_max_body_bytes=2048,
    )
    switchboard = Switchboard({"fx.test": fx, "batch.test": batch})
    upstream = httpx.AsyncClient(transport=switchboard)
    directory = StaticDirectory(
        ["http://fx.test", "http://batch.test"], http=upstream, ttl_seconds=3600
    )
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(fx.router.lifespan_context(fx))
        await stack.enter_async_context(batch.router.lifespan_context(batch))
        await directory.agents()
        switchboard.down.update(f"{n}.test" for n in down)
        app = create_gateway_app(
            gw_settings,
            directory,
            public_url="http://gw.test",
            http_client=upstream,
            rate_limiter=InMemoryRateLimiter(limit),
            catalog_ttl_seconds=3600,
        )
        await stack.enter_async_context(app.router.lifespan_context(app))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
        ) as client:
            yield client
        await upstream.aclose()


async def test_agent_card_is_rewritten_to_point_at_the_gateway():
    async with gateway() as gw:
        card = (
            await gw.get(
                "/agents/fx-conversion-agent/.well-known/agent-card.json",
                headers={"x-api-key": "alice-key"},
            )
        ).json()
    urls = {i["protocolBinding"]: i["url"] for i in card["supportedInterfaces"]}
    assert urls == {
        "JSONRPC": f"{FX_URL}/a2a/jsonrpc",
        "HTTP+JSON": f"{FX_URL}/a2a/rest",
    }
    assert card["name"] == "FX Conversion Agent"


@pytest.mark.parametrize("binding", ["JSONRPC", "HTTP+JSON"])
@pytest.mark.parametrize("streaming", [True, False])
async def test_clients_use_agents_through_the_gateway(binding, streaming):
    async with gateway() as gw:
        client = await AgentClient.connect(
            FX_URL, http=gw, api_key="alice-key", binding=binding, streaming=streaming
        )
        first = await client.send("convert 100 to EUR")
        assert first.needs_input
        done = await client.send("USD", task_id=first.task_id, context_id=first.context_id)
    assert done.succeeded
    assert "90.00 EUR" in done.text


async def test_gateway_authenticates_clients_and_swaps_in_upstream_credentials():
    async with gateway() as gw:
        anonymous = await gw.get("/agents/fx-conversion-agent/.well-known/agent-card.json")
        wrong = await gw.get("/agents", headers={"x-api-key": UPSTREAM_KEY})
        ok = await gw.get("/agents", headers={"x-api-key": "bob-key"})
        health = await gw.get("/healthz")
    assert anonymous.status_code == 401
    assert wrong.status_code == 401  # the agents' own key is not a gateway credential
    assert health.status_code == 200
    listing = ok.json()["agents"]
    assert [a["id"] for a in listing] == ["batch-processing-agent", "fx-conversion-agent"]
    assert listing[0]["card"].startswith("http://gw.test/agents/")


async def test_rate_limit_is_enforced_per_principal():
    async with gateway(limit=3) as gw:
        alice = {"x-api-key": "alice-key"}
        statuses = [
            (
                await gw.get(
                    "/agents/fx-conversion-agent/.well-known/agent-card.json", headers=alice
                )
            ).status_code
            for _ in range(4)
        ]
        limited = await gw.get(
            "/agents/fx-conversion-agent/.well-known/agent-card.json", headers=alice
        )
        bob = await gw.get(
            "/agents/fx-conversion-agent/.well-known/agent-card.json",
            headers={"x-api-key": "bob-key"},
        )
    assert statuses == [200, 200, 200, 429]
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.headers["x-ratelimit-remaining"] == "0"
    assert bob.status_code == 200
    assert bob.headers["x-ratelimit-limit"] == "3"


async def test_unknown_agents_and_oversized_bodies_are_rejected():
    async with gateway() as gw:
        headers = {"x-api-key": "alice-key"}
        missing = await gw.get("/agents/ghost/.well-known/agent-card.json", headers=headers)
        huge = await gw.post(
            "/agents/fx-conversion-agent/a2a/jsonrpc", content=b"x" * 5000, headers=headers
        )
    assert missing.status_code == 404
    assert huge.status_code == 413


async def test_unreachable_upstream_maps_to_502_then_opens_the_circuit():
    async with gateway(down=("fx",)) as gw:
        headers = {"x-api-key": "alice-key"}
        path = "/agents/fx-conversion-agent/a2a/jsonrpc"
        first = [(await gw.post(path, json={}, headers=headers)).status_code for _ in range(5)]
        tripped = await gw.post(path, json={}, headers=headers)
        other = await gw.get(
            "/agents/batch-processing-agent/.well-known/agent-card.json", headers=headers
        )
    assert first == [502] * 5
    assert tripped.status_code == 503
    assert tripped.headers["retry-after"] == "15"
    assert other.status_code == 200  # other agents are unaffected


async def test_in_memory_limiter_refills():
    limiter = InMemoryRateLimiter(60)  # one token per second
    decisions = [await limiter.check("k") for _ in range(61)]
    assert all(d.allowed for d in decisions[:60])
    assert not decisions[60].allowed
    assert decisions[60].retry_after_seconds > 0


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds


async def test_redis_limiter_counts_per_window():
    clock = [1000.0]
    redis = FakeRedis()
    limiter = RedisFixedWindowLimiter(redis, 2, clock=lambda: clock[0])

    assert (await limiter.check("alice")).allowed
    second = await limiter.check("alice")
    assert second.allowed and second.remaining == 0
    blocked = await limiter.check("alice")
    assert not blocked.allowed
    assert 0 < blocked.retry_after_seconds <= 60
    assert (await limiter.check("bob")).allowed  # independent key
    assert all(ttl == 61 for ttl in redis.ttls.values())

    clock[0] += 60  # next window
    assert (await limiter.check("alice")).allowed
