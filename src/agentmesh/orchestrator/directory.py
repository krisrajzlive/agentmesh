"""Where the orchestrator learns which agents exist."""

from __future__ import annotations

import time
from typing import Protocol

import httpx
from a2a.client import A2ACardResolver

from agentmesh.observability.logging import get_logger
from agentmesh.registry.client import RegistryClient
from agentmesh.registry.models import AgentRecord, HealthStatus, record_from_card

log = get_logger(__name__)


class AgentDirectory(Protocol):
    async def agents(self) -> list[AgentRecord]:
        """Healthy agents that can currently receive work."""
        ...

    async def aclose(self) -> None: ...


class RegistryDirectory:
    """Backed by the registry service."""

    def __init__(self, client: RegistryClient) -> None:
        self._client = client

    async def agents(self) -> list[AgentRecord]:
        return await self._client.list(status=HealthStatus.HEALTHY.value)

    async def aclose(self) -> None:
        await self._client.close()


class StaticDirectory:
    """A fixed list of agent URLs whose cards are fetched lazily and cached."""

    def __init__(
        self, urls: list[str], *, http: httpx.AsyncClient | None = None, ttl_seconds: float = 60.0
    ) -> None:
        self._urls = urls
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._ttl = ttl_seconds
        self._cache: tuple[float, list[AgentRecord]] | None = None

    async def agents(self) -> list[AgentRecord]:
        if self._cache and time.monotonic() - self._cache[0] < self._ttl:
            return self._cache[1]
        records: list[AgentRecord] = []
        for url in self._urls:
            try:
                card = await A2ACardResolver(self._http, url).get_agent_card()
            except Exception as exc:
                log.warning("static_agent_unreachable", url=url, error=str(exc))
                continue
            record = record_from_card(url, card)
            record.status = HealthStatus.HEALTHY
            records.append(record)
        self._cache = (time.monotonic(), records)
        return records

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
