"""HTTP API for the agent registry."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agentmesh.config import Settings
from agentmesh.observability.logging import get_logger
from agentmesh.registry.models import HealthStatus
from agentmesh.registry.service import (
    ConflictError,
    NotFoundError,
    RegistrationError,
    RegistryService,
)
from agentmesh.registry.store import RegistryStore, build_store
from agentmesh.runtime.middleware import AuthMiddleware, RequestContextMiddleware
from agentmesh.security.auth import build_authenticator

log = get_logger(__name__)

PUBLIC_PATHS = ("/healthz", "/readyz", "/metrics")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def create_registry_app(
    settings: Settings,
    *,
    store: RegistryStore | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Starlette:
    store = store or build_store(settings.registry_database_url)
    http = http_client or httpx.AsyncClient(timeout=10.0)
    service = RegistryService(
        store,
        http,
        allow_private_urls=settings.registry_allow_private_urls,
        unhealthy_after=settings.registry_unhealthy_after,
    )
    state = {"ready": False}

    async def register(request: Request) -> Response:
        try:
            body = await request.json()
            url = body["url"]
            if not isinstance(url, str):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return _error(400, "invalid_request", 'body must be {"url": "<agent base url>"}')
        try:
            record = await service.register(url)
        except RegistrationError as exc:
            return _error(422, "registration_failed", str(exc))
        except ConflictError as exc:
            return _error(409, "conflict", str(exc))
        return JSONResponse(record.model_dump(mode="json"), status_code=201)

    async def list_agents(request: Request) -> Response:
        params = request.query_params
        status = params.get("status")
        try:
            status_filter = HealthStatus(status) if status else None
        except ValueError:
            return _error(400, "invalid_request", f"unknown status {status!r}")
        records = await service.search(
            skill=params.get("skill"),
            tag=params.get("tag"),
            query=params.get("q"),
            status=status_filter,
        )
        return JSONResponse(
            {"agents": [r.model_dump(mode="json", exclude={"card"}) for r in records]}
        )

    async def get_agent(request: Request) -> Response:
        try:
            record = await service.get(request.path_params["agent_id"])
        except NotFoundError:
            return _error(404, "not_found", "no such agent")
        return JSONResponse(record.model_dump(mode="json"))

    async def delete_agent(request: Request) -> Response:
        try:
            await service.deregister(request.path_params["agent_id"])
        except NotFoundError:
            return _error(404, "not_found", "no such agent")
        return Response(status_code=204)

    async def check_agent(request: Request) -> Response:
        try:
            record = await service.check(request.path_params["agent_id"])
        except NotFoundError:
            return _error(404, "not_found", "no such agent")
        return JSONResponse(record.model_dump(mode="json", exclude={"card"}))

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz(_: Request) -> Response:
        return JSONResponse({"ready": state["ready"]}, status_code=200 if state["ready"] else 503)

    async def metrics(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await store.start()
        for url in settings.static_agent_urls:
            try:
                await service.register(url)
            except (RegistrationError, ConflictError) as exc:
                log.warning("seed_registration_failed", url=url, error=str(exc))
        loop_task = asyncio.create_task(
            service.run_health_loop(settings.registry_health_interval_seconds)
        )
        state["ready"] = True
        log.info("registry_started")
        try:
            yield
        finally:
            state["ready"] = False
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
            if http_client is None:
                await http.aclose()
            await store.close()

    app = Starlette(
        routes=[
            Route("/v1/agents", register, methods=["POST"]),
            Route("/v1/agents", list_agents, methods=["GET"]),
            Route("/v1/agents/{agent_id}", get_agent, methods=["GET"]),
            Route("/v1/agents/{agent_id}", delete_agent, methods=["DELETE"]),
            Route("/v1/agents/{agent_id}/check", check_agent, methods=["POST"]),
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/metrics", metrics),
        ],
        lifespan=lifespan,
        middleware=[
            Middleware(RequestContextMiddleware, component="registry"),
            Middleware(
                AuthMiddleware,
                authenticator=build_authenticator(settings),
                public_paths=PUBLIC_PATHS,
            ),
        ],
    )
    app.state.service = service
    return app
