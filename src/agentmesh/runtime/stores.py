"""Task and push-config persistence selected by ``AGENTMESH_TASK_STORE_URL``."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def build_persistence(settings: Settings) -> Persistence:
    """``memory`` keeps state in-process; any SQLAlchemy async URL persists it."""
    url = settings.task_store_url
    if url == "memory":
        return Persistence(InMemoryTaskStore(), InMemoryPushNotificationConfigStore())
    engine = create_async_engine(url, pool_pre_ping=True)
    return Persistence(
        task_store=DatabaseTaskStore(engine),
        push_config_store=DatabasePushNotificationConfigStore(engine),
        engine=engine,
    )
