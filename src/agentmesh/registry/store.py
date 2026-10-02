"""Registry persistence: in-memory and SQL (SQLite/PostgreSQL) implementations."""

from __future__ import annotations

import asyncio
from typing import Protocol

from sqlalchemy import Column, MetaData, String, Table, Text, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from agentmesh.registry.models import AgentRecord


class RegistryStore(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def upsert(self, record: AgentRecord) -> None: ...
    async def get(self, agent_id: str) -> AgentRecord | None: ...
    async def list(self) -> list[AgentRecord]: ...
    async def delete(self, agent_id: str) -> bool: ...


class InMemoryRegistryStore:
    def __init__(self) -> None:
        self._records: dict[str, AgentRecord] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert(self, record: AgentRecord) -> None:
        async with self._lock:
            self._records[record.id] = record.model_copy(deep=True)

    async def get(self, agent_id: str) -> AgentRecord | None:
        record = self._records.get(agent_id)
        return record.model_copy(deep=True) if record else None

    async def list(self) -> list[AgentRecord]:
        return [r.model_copy(deep=True) for r in self._records.values()]

    async def delete(self, agent_id: str) -> bool:
        async with self._lock:
            return self._records.pop(agent_id, None) is not None


_metadata = MetaData()
_agents = Table(
    "registry_agents",
    _metadata,
    Column("id", String(128), primary_key=True),
    Column("record", Text, nullable=False),
)


class SqlRegistryStore:
    """Stores each record as a JSON document keyed by agent id."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> SqlRegistryStore:
        if ":memory:" in url:
            # A single shared connection keeps the in-memory database alive.
            engine = create_async_engine(
                url, poolclass=StaticPool, connect_args={"check_same_thread": False}
            )
        else:
            engine = create_async_engine(url, pool_pre_ping=True)
        return cls(engine)

    async def start(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def upsert(self, record: AgentRecord) -> None:
        values = {"id": record.id, "record": record.model_dump_json()}
        dialect = self._engine.dialect.name
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        statement = insert(_agents).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[_agents.c.id], set_={"record": statement.excluded.record}
        )
        async with self._engine.begin() as conn:
            await conn.execute(statement)

    async def get(self, agent_id: str) -> AgentRecord | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(select(_agents.c.record).where(_agents.c.id == agent_id))
            ).first()
        return AgentRecord.model_validate_json(row[0]) if row else None

    async def list(self) -> list[AgentRecord]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(select(_agents.c.record))).all()
        return [AgentRecord.model_validate_json(r[0]) for r in rows]

    async def delete(self, agent_id: str) -> bool:
        async with self._engine.begin() as conn:
            result = await conn.execute(delete(_agents).where(_agents.c.id == agent_id))
        return result.rowcount > 0


def build_store(database_url: str) -> RegistryStore:
    if database_url == "memory":
        return InMemoryRegistryStore()
    return SqlRegistryStore.from_url(database_url)
