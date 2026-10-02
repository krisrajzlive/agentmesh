"""API gateway: one authenticated, rate-limited entry point in front of every agent."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agentmesh.config import Settings
from agentmesh.gateway.ratelimit import RateLimiter, build_rate_limiter
from agentmesh.observability.logging import get_logger
from agentmesh.orchestrator.directory import AgentDirectory
from agentmesh.registry.models import AgentRecord
from agentmesh.resilience import CircuitBreaker
from agentmesh.runtime.middleware import AuthMiddleware, RequestContextMiddleware
from agentmesh.security.auth import build_authenticator

log = get_logger(__name__)

UPSTREAM = Counter(
    "agentmesh_gateway_upstream_total", "Requests proxied to agents", ["agent", "outcome"]
)
PUBLIC_PATHS = ("/healthz", "/readyz", "/metrics")

# Request headers that are safe and useful to forward. Everything else (notably the caller's
# own credentials and cookies) is dropped; the gateway attaches its own upstream credentials.
FORWARD_HEADERS = frozenset(
    {"content-type", "accept", "a2a-version", "a2a-extensions", "last-event-id", "x-request-id"}
)
RESPONSE_HEADERS = frozenset(
    {"content-type", "cache-control", "etag", "retry-after", "a2a-version"}
)


def _error(status: int, code: str, message: str, **headers: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


def _rewrite_card(card: dict[str, Any], public_base: str, slug: str) -> dict[str, Any]:
    """Point every advertised interface at the gateway so clients never bypass it."""
    for interface in card.get("supportedInterfaces", []):
        path = httpx.URL(interface["url"]).path
        interface["url"] = f"{public_base}/agents/{slug}{path}"
    return card


def create_gateway_app(
    settings: Settings,
    directory: AgentDirectory,
    *,
    public_url: str,
    http_client: httpx.AsyncClient | None = None,
    rate_limiter: RateLimiter | None = None,
    catalog_ttl_seconds: float = 5.0,
) -> Starlette:
    public_base = public_url.rstrip("/")
    http = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
    )
    limiter = rate_limiter or build_rate_limiter(
        settings.gateway_rate_limit_per_minute, settings.gateway_redis_url
    )
    breakers: dict[str, CircuitBreaker] = {}
    catalog: dict[str, Any] = {"at": 0.0, "agents": {}}
    state = {"ready": False}

    upstream_headers: dict[str, str] = {}
    if settings.outbound_api_key:
        upstream_headers["x-api-key"] = settings.outbound_api_key.get_secret_value()
    if settings.outbound_bearer_token:
        upstream_headers["authorization"] = (
            f"Bearer {settings.outbound_bearer_token.get_secret_value()}"
        )

    async def agents_by_id() -> dict[str, AgentRecord]:
        if time.monotonic() - catalog["at"] > catalog_ttl_seconds:
            catalog["agents"] = {a.id: a for a in await directory.agents()}
            catalog["at"] = time.monotonic()
        agents: dict[str, AgentRecord] = catalog["agents"]
        return agents

    async def list_agents(request: Request) -> Response:
        agents = await agents_by_id()
        return JSONResponse(
            {
                "agents": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "framework": a.framework,
                        "skills": [s.id for s in a.skills],
                        "card": f"{public_base}/agents/{a.id}{AGENT_CARD_WELL_KNOWN_PATH}",
                    }
                    for a in sorted(agents.values(), key=lambda a: a.id)
                ]
            }
        )

    async def proxy(request: Request) -> Response:
        slug = request.path_params["slug"]
        path = "/" + request.path_params.get("path", "")
        agent = (await agents_by_id()).get(slug)
        if agent is None:
            return _error(404, "unknown_agent", f"no agent {slug!r} is available")

        principal = request.user.display_name if "user" in request.scope else "anonymous"
        decision = await limiter.check(principal)
        rate_headers = {
            "x-ratelimit-limit": str(decision.limit),
            "x-ratelimit-remaining": str(decision.remaining),
        }
        if not decision.allowed:
            UPSTREAM.labels(slug, "rate_limited").inc()
            return _error(
                429,
                "rate_limited",
                "too many requests",
                **{"retry-after": str(max(1, round(decision.retry_after_seconds))), **rate_headers},
            )

        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.gateway_max_body_bytes:
            return _error(413, "payload_too_large", "request body exceeds the gateway limit")
        body = await request.body()
        if len(body) > settings.gateway_max_body_bytes:
            return _error(413, "payload_too_large", "request body exceeds the gateway limit")

        breaker = breakers.setdefault(
            agent.url, CircuitBreaker(f"gateway:{slug}", failure_threshold=5, reset_timeout=15)
        )
        if not breaker.allow():
            UPSTREAM.labels(slug, "circuit_open").inc()
            return _error(
                503,
                "upstream_unavailable",
                "agent is temporarily unavailable",
                **{"retry-after": "15"},
            )

        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        if request_id:
            headers["x-request-id"] = str(request_id)
        headers.update(upstream_headers)

        target = f"{agent.url}{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        upstream_request = http.build_request(request.method, target, headers=headers, content=body)

        try:
            upstream = await http.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            breaker.record_failure()
            UPSTREAM.labels(slug, "timeout").inc()
            return _error(504, "upstream_timeout", "agent did not respond in time")
        except httpx.TransportError as exc:
            breaker.record_failure()
            UPSTREAM.labels(slug, "unreachable").inc()
            log.warning("gateway_upstream_unreachable", agent=slug, error=str(exc))
            return _error(502, "upstream_unreachable", "agent is unreachable")

        if upstream.status_code >= 500:
            breaker.record_failure()
        else:
            breaker.record_success()
        UPSTREAM.labels(slug, "ok" if upstream.status_code < 500 else "error").inc()

        response_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() in RESPONSE_HEADERS
        }
        response_headers.update(rate_headers)
        if path == AGENT_CARD_WELL_KNOWN_PATH and upstream.status_code == 200:
            card = _rewrite_card(json.loads(await upstream.aread()), public_base, slug)
            await upstream.aclose()
            response_headers.pop(
                "etag", None
            )  # body changed, so the upstream tag no longer applies
            return JSONResponse(card, headers=response_headers)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz(_: Request) -> Response:
        return JSONResponse({"ready": state["ready"]}, status_code=200 if state["ready"] else 503)

    async def metrics(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        state["ready"] = True
        log.info("gateway_started", url=public_base)
        try:
            yield
        finally:
            state["ready"] = False
            await directory.aclose()
            if http_client is None:
                await http.aclose()

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    return Starlette(
        routes=[
            Route("/agents", list_agents, methods=["GET"]),
            Route("/agents/{slug}/{path:path}", proxy, methods=methods),
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/metrics", metrics),
        ],
        lifespan=lifespan,
        middleware=[
            Middleware(RequestContextMiddleware, component="gateway"),
            Middleware(
                AuthMiddleware,
                authenticator=build_authenticator(settings),
                public_paths=PUBLIC_PATHS,
            ),
        ],
    )
