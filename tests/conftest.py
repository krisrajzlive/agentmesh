from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from a2a.client import ClientConfig, create_client
from a2a.client.client import Client
from starlette.applications import Starlette

from agentmesh.config import Settings

BASE_URL = "http://agent.test"


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, environment="development", auth_mode="none")  # type: ignore[call-arg]


@asynccontextmanager
async def serve(app: Starlette) -> AsyncIterator[httpx.AsyncClient]:
    """Run ``app`` in-process (lifespan included) and yield an httpx client bound to it."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as http:
            yield http


async def a2a_client(
    http: httpx.AsyncClient,
    *,
    binding: str = "JSONRPC",
    streaming: bool = True,
    headers: dict[str, str] | None = None,
) -> Client:
    if headers:
        http.headers.update(headers)
    config = ClientConfig(
        httpx_client=http, streaming=streaming, supported_protocol_bindings=[binding]
    )
    return await create_client(BASE_URL, client_config=config)


@pytest.fixture
def serve_app() -> Callable[[Starlette], Any]:
    return serve
