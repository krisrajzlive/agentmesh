"""Pure-ASGI middleware (safe for SSE streaming, unlike BaseHTTPMiddleware)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Collection

import structlog
from starlette.authentication import AuthCredentials, SimpleUser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentmesh.observability.logging import get_logger
from agentmesh.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from agentmesh.security.auth import Authenticator

log = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"


class RequestContextMiddleware:
    """Assigns a request id, binds it to the log context, records metrics and access logs."""

    def __init__(self, app: ASGIApp, *, component: str) -> None:
        self.app = app
        self.component = component

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        request_id = headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, component=self.component)

        status_holder = {"status": 500}
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            path = scope["path"]
            HTTP_REQUESTS.labels(
                self.component, scope["method"], path, status_holder["status"]
            ).inc()
            HTTP_LATENCY.labels(self.component, path).observe(elapsed)
            log.info(
                "http_request",
                method=scope["method"],
                path=path,
                status=status_holder["status"],
                duration_ms=round(elapsed * 1000, 1),
            )


class AuthMiddleware:
    """Rejects unauthenticated requests and exposes the caller to the A2A call context."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        authenticator: Authenticator | None,
        public_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self.authenticator = authenticator
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or self.authenticator is None
            or scope["path"] in self.public_paths
        ):
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        principal = self.authenticator.authenticate(headers)
        if principal is None:
            log.warning("auth_rejected", path=scope["path"])
            body = json.dumps(
                {"error": "unauthorized", "message": "valid credentials required"}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer realm="agentmesh"'),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        scope["user"] = SimpleUser(principal.name)
        scope["auth"] = AuthCredentials(list(principal.scopes))
        structlog.contextvars.bind_contextvars(principal=principal.name, auth=principal.method)
        await self.app(scope, receive, send)
