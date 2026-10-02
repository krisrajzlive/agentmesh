"""Registry business logic: registration, discovery and health tracking."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import httpx
from a2a.client import A2ACardResolver
from a2a.client.errors import A2AClientError

from agentmesh.observability.logging import get_logger
from agentmesh.registry.models import AgentRecord, HealthStatus, record_from_card
from agentmesh.registry.store import RegistryStore
from agentmesh.security.ssrf import UnsafeURLError, validate_outbound_url

log = get_logger(__name__)


class RegistrationError(Exception):
    """The agent could not be registered (unreachable, invalid card, unsafe URL)."""


class ConflictError(Exception):
    """A different agent already uses this name."""


class NotFoundError(KeyError):
    pass


class RegistryService:
    def __init__(
        self,
        store: RegistryStore,
        http: httpx.AsyncClient,
        *,
        allow_private_urls: bool = True,
        unhealthy_after: int = 3,
    ) -> None:
        self._store = store
        self._http = http
        self._allow_private = allow_private_urls
        self._unhealthy_after = unhealthy_after

    async def _fetch(self, url: str):  # type: ignore[no-untyped-def]
        try:
            await validate_outbound_url(url, allow_private=self._allow_private)
            return await A2ACardResolver(self._http, url).get_agent_card()
        except UnsafeURLError as exc:
            raise RegistrationError(f"refusing to contact {url}: {exc}") from exc
        except (A2AClientError, httpx.HTTPError, ValueError) as exc:
            raise RegistrationError(
                f"could not fetch a valid Agent Card from {url}: {exc}"
            ) from exc

    async def register(self, url: str) -> AgentRecord:
        card = await self._fetch(url)
        if not card.skills:
            raise RegistrationError("agent declares no skills")
        record = record_from_card(url, card)
        existing = await self._store.get(record.id)
        if existing is not None:
            if existing.url != record.url:
                raise ConflictError(
                    f"agent id {record.id!r} is already registered at {existing.url}"
                )
            record.registered_at = existing.registered_at
        record.status = HealthStatus.HEALTHY
        record.last_checked_at = datetime.now(UTC)
        await self._store.upsert(record)
        log.info("agent_registered", agent=record.id, url=record.url, framework=record.framework)
        return record

    async def deregister(self, agent_id: str) -> None:
        if not await self._store.delete(agent_id):
            raise NotFoundError(agent_id)
        log.info("agent_deregistered", agent=agent_id)

    async def get(self, agent_id: str) -> AgentRecord:
        record = await self._store.get(agent_id)
        if record is None:
            raise NotFoundError(agent_id)
        return record

    async def search(
        self,
        *,
        skill: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        status: HealthStatus | None = None,
    ) -> list[AgentRecord]:
        records = await self._store.list()
        matched = [
            r
            for r in records
            if r.matches(skill=skill, tag=tag, query=query)
            and (status is None or r.status == status)
        ]
        return sorted(matched, key=lambda r: r.id)

    async def check(self, agent_id: str) -> AgentRecord:
        record = await self.get(agent_id)
        return await self._check_record(record)

    async def _check_record(self, record: AgentRecord) -> AgentRecord:
        now = datetime.now(UTC)
        try:
            card = await self._fetch(record.url)
        except RegistrationError as exc:
            record.consecutive_failures += 1
            record.last_error = str(exc)
            if record.consecutive_failures >= self._unhealthy_after:
                record.status = HealthStatus.UNHEALTHY
            log.warning("agent_check_failed", agent=record.id, failures=record.consecutive_failures)
        else:
            refreshed = record_from_card(record.url, card, agent_id=record.id)
            refreshed.registered_at = record.registered_at
            refreshed.status = HealthStatus.HEALTHY
            record = refreshed
        record.last_checked_at = now
        await self._store.upsert(record)
        return record

    async def check_all(self) -> None:
        records = await self._store.list()
        await asyncio.gather(*(self._check_record(r) for r in records))

    async def run_health_loop(self, interval_seconds: float) -> None:
        """Periodically refresh every agent. Runs until cancelled."""
        while True:
            await asyncio.sleep(interval_seconds)
            with contextlib.suppress(Exception):
                await self.check_all()
