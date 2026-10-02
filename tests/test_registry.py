from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal

import httpx
import pytest

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.config import Settings
from agentmesh.registry import (
    HealthStatus,
    RegistryClient,
    RegistryClientError,
    SqlRegistryStore,
    create_registry_app,
)
from agentmesh.registry.models import AgentRecord
from agentmesh.runtime import create_agent_app
from agentmesh.security.ssrf import UnsafeURLError, validate_outbound_url


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _skip_dns(monkeypatch, request):
    """In-process hosts like ``fx.test`` do not resolve; SSRF logic has its own tests."""
    if "real_ssrf" in request.keywords:
        return

    async def allow(url: str, *, allow_private: bool) -> None:
        return None

    monkeypatch.setattr("agentmesh.registry.service.validate_outbound_url", allow)


@asynccontextmanager
async def platform(*, settings: Settings | None = None, store=None):
    """A registry plus two in-process agents it can reach by hostname."""
    settings = settings or make_settings()
    fx = create_agent_app(
        FX_SPEC,
        build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")})),
        settings,
        public_url="http://fx.test",
    )
    batch = create_agent_app(BATCH_SPEC, BatchExecutor(), settings, public_url="http://batch.test")
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(fx.router.lifespan_context(fx))
        await stack.enter_async_context(batch.router.lifespan_context(batch))
        outbound = httpx.AsyncClient(
            mounts={
                "http://fx.test": httpx.ASGITransport(app=fx),
                "http://batch.test": httpx.ASGITransport(app=batch),
            }
        )
        registry = create_registry_app(settings, store=store, http_client=outbound)
        await stack.enter_async_context(registry.router.lifespan_context(registry))
        api = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=registry), base_url="http://registry.test"
        )
        client = RegistryClient("http://registry.test", http=api)
        yield client, registry
        await api.aclose()
        await outbound.aclose()


async def test_register_and_discover_agents_across_frameworks():
    async with platform() as (registry, _):
        fx = await registry.register("http://fx.test")
        await registry.register("http://batch.test")

        assert fx.id == "fx-conversion-agent"
        assert fx.status is HealthStatus.HEALTHY
        assert fx.framework == "agentmesh"
        assert fx.skills[0].id == "currency-conversion"

        assert [a.id for a in await registry.list()] == [
            "batch-processing-agent",
            "fx-conversion-agent",
        ]
        assert [a.id for a in await registry.list(skill="batch-hash")] == ["batch-processing-agent"]
        assert [a.id for a in await registry.list(tag="finance")] == ["fx-conversion-agent"]
        assert [a.id for a in await registry.list(query="currency")] == ["fx-conversion-agent"]
        assert await registry.list(skill="nope") == []


async def test_registration_is_idempotent_and_get_delete_work():
    async with platform() as (registry, _):
        first = await registry.register("http://fx.test")
        again = await registry.register("http://fx.test/")
        assert again.registered_at == first.registered_at
        assert len(await registry.list()) == 1

        assert (await registry.get(first.id)).url == "http://fx.test"
        await registry.deregister(first.id)
        with pytest.raises(RegistryClientError, match="404"):
            await registry.get(first.id)


async def test_registration_failures_are_reported():
    async with platform() as (registry, _):
        with pytest.raises(RegistryClientError, match="422"):
            await registry.register("http://missing.test")
        with pytest.raises(RegistryClientError, match="400|422"):
            await registry._call("POST", "/v1/agents", json={"nope": 1})


async def test_health_check_marks_unreachable_agents_unhealthy():
    settings = make_settings(registry_unhealthy_after=2)
    async with platform(settings=settings) as (registry, app):
        record = await registry.register("http://fx.test")
        service = app.state.service
        # Point the record at a dead host, then probe it repeatedly.
        stored = await service._store.get(record.id)
        stored.url = "http://dead.test"
        await service._store.upsert(stored)

        first = await service.check(record.id)
        assert first.status is HealthStatus.HEALTHY and first.consecutive_failures == 1
        second = await service.check(record.id)
        assert second.status is HealthStatus.UNHEALTHY
        assert second.last_error

        assert [a.id for a in await registry.list(status="unhealthy")] == [record.id]

        # Recovery: once reachable again the agent is healthy and counters reset.
        second.url = "http://fx.test"
        await service._store.upsert(second)
        healed = await service.check(record.id)
        assert healed.status is HealthStatus.HEALTHY and healed.consecutive_failures == 0


async def test_registry_requires_authentication_when_enabled():
    settings = make_settings(auth_mode="api_key", api_keys="admin:adm-key")
    async with platform(settings=settings) as (_, app):
        anonymous = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://r.test"
        )
        assert (await anonymous.get("/v1/agents")).status_code == 401
        assert (await anonymous.get("/healthz")).status_code == 200
        ok = RegistryClient("http://r.test", api_key="adm-key", http=anonymous)
        assert await ok.list() == []


async def test_sql_store_persists_records(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    settings = make_settings(registry_database_url=url)

    async with platform(settings=settings, store=SqlRegistryStore.from_url(url)) as (registry, _):
        await registry.register("http://fx.test")

    reopened = SqlRegistryStore.from_url(url)
    await reopened.start()
    records = await reopened.list()
    assert [r.id for r in records] == ["fx-conversion-agent"]
    assert isinstance(records[0], AgentRecord)
    assert await reopened.delete("fx-conversion-agent") is True
    assert await reopened.get("fx-conversion-agent") is None
    await reopened.close()


@pytest.mark.real_ssrf
async def test_ssrf_validation_blocks_dangerous_targets():
    for url in (
        "http://169.254.169.254/latest/meta-data",  # cloud metadata, blocked even when private is allowed
        "ftp://example.com/",
        "http://user:pw@10.0.0.1/",
        "not a url",
    ):
        with pytest.raises(UnsafeURLError):
            await validate_outbound_url(url, allow_private=True)

    with pytest.raises(UnsafeURLError):
        await validate_outbound_url("http://127.0.0.1:8080", allow_private=False)
    with pytest.raises(UnsafeURLError):
        await validate_outbound_url("http://10.1.2.3", allow_private=False)
    with pytest.raises(UnsafeURLError):
        await validate_outbound_url("http://[::ffff:169.254.169.254]/", allow_private=True)
    with pytest.raises(UnsafeURLError):
        await validate_outbound_url("http://[::1]:9000", allow_private=False)
    await validate_outbound_url("http://[::1]:9000", allow_private=True)
    await validate_outbound_url("http://10.1.2.3", allow_private=True)  # in-cluster agents
    await validate_outbound_url("http://127.0.0.1:9001", allow_private=True)
