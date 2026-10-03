"""Task and push-config persistence selected by ``AGENTMESH_TASK_STORE_URL``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from a2a.server.models import Base
from a2a.server.tasks import (
    DatabasePushNotificationConfigStore,
    DatabaseTaskStore,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    PushNotificationConfigStore,
    TaskStore,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from agentmesh.config import Settings


@dataclass
class Persistence:
    task_store: TaskStore
    push_config_store: PushNotificationConfigStore
    engine: AsyncEngine | None = None
    _initializers: list[object] = field(default_factory=list)

    async def start(self) -> None:
        for store in (self.task_store, self.push_config_store):
            initialize = getattr(store, "initialize", None)
            if initialize is not None:
                await initialize()

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()


def _table_prefix(namespace: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", namespace.lower()).strip("_") or "agent"


def _release_table(name: str) -> None:
    """Forget a table the SDK declared earlier in this process.

    The SDK declares each custom-named model on a global ``Base``, so building a store twice in
    one process (tests, hot reload) would raise "Table already defined".
    """
    table = Base.metadata.tables.get(name)
    if table is not None:
        Base.metadata.remove(table)


def build_persistence(settings: Settings, namespace: str = "agent") -> Persistence:
    """``memory`` keeps state in-process; any SQLAlchemy async URL persists it.

    ``namespace`` (the agent slug) prefixes the table names, so several agents can share one
    database without seeing each other's tasks.
    """
    url = settings.task_store_url
    if url == "memory":
        return Persistence(InMemoryTaskStore(), InMemoryPushNotificationConfigStore())
    engine = create_async_engine(url, pool_pre_ping=True)
    prefix = _table_prefix(namespace)
    _release_table(f"{prefix}_tasks")
    _release_table(f"{prefix}_push_configs")
    return Persistence(
        task_store=DatabaseTaskStore(engine, table_name=f"{prefix}_tasks"),
        push_config_store=DatabasePushNotificationConfigStore(
            engine, table_name=f"{prefix}_push_configs"
        ),
        engine=engine,
    )
