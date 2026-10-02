"""Assemble a production A2A server (Starlette ASGI app) around any ``AgentExecutor``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.utils.push_url_validator import validate_push_notification_url
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agentmesh.config import Settings
from agentmesh.observability.logging import get_logger
from agentmesh.runtime.middleware import AuthMiddleware, RequestContextMiddleware
from agentmesh.runtime.push import SignedPushNotificationSender
from agentmesh.runtime.spec import JSONRPC_PATH, REST_PATH, AgentSpec, build_agent_card
from agentmesh.runtime.stores import Persistence, build_persistence
from agentmesh.security.auth import build_authenticator

log = get_logger(__name__)

PUBLIC_PATHS = (AGENT_CARD_WELL_KNOWN_PATH, "/healthz", "/readyz", "/metrics")


async def _allow_any(_: str) -> bool:
    return True


def create_agent_app(
    spec: AgentSpec,
    executor: AgentExecutor,
    settings: Settings,
    *,
    public_url: str,
    persistence: Persistence | None = None,
) -> Starlette:
    """Build the ASGI app serving ``spec`` over JSON-RPC and HTTP+JSON, with ops endpoints."""
    persistence = persistence or build_persistence(settings)
    http_client = httpx.AsyncClient(timeout=10.0)
    validator = _allow_any if settings.allow_private_push_urls else validate_push_notification_url
    sender = SignedPushNotificationSender(
        http_client,
        persistence.push_config_store,
        signing_secret=(
            settings.push_signing_secret.get_secret_value()
            if settings.push_signing_secret
            else None
        ),
        push_url_validator=validator,
    )

    card = build_agent_card(spec, public_url, settings)
    extended_card = build_agent_card(spec, public_url, settings, extended=True)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=persistence.task_store,
        agent_card=card,
        push_config_store=persistence.push_config_store,
        push_sender=sender,
        extended_agent_card=extended_card if spec.extended_skills else None,
        push_url_validator=validator,
        validate_input_modes=True,
    )

    state = {"ready": False}

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok", "agent": spec.slug})

    async def readyz(_: Request) -> Response:
        code = 200 if state["ready"] else 503
        return JSONResponse({"ready": state["ready"], "agent": spec.slug}, status_code=code)

    async def metrics(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await persistence.start()
        state["ready"] = True
        log.info("agent_started", agent=spec.slug, framework=spec.framework, url=public_url)
        try:
            yield
        finally:
            state["ready"] = False
            await sender.drain()
            await http_client.aclose()
            await persistence.close()
            log.info("agent_stopped", agent=spec.slug)

    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, JSONRPC_PATH),
        *create_rest_routes(handler, path_prefix=REST_PATH),
        Route("/healthz", healthz),
        Route("/readyz", readyz),
        Route("/metrics", metrics),
    ]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            Middleware(RequestContextMiddleware, component=spec.slug),
            Middleware(
                AuthMiddleware,
                authenticator=build_authenticator(settings),
                public_paths=PUBLIC_PATHS,
            ),
        ],
    )
    app.state.spec = spec
    app.state.card = card
    app.state.push_sender = sender
    return app


__all__ = ["JSONRPC_PATH", "REST_PATH", "create_agent_app"]
