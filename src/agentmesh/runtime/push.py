"""Push-notification delivery: HMAC-signed payloads, retries and a dead-letter queue."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from a2a.server.tasks import (
    BasePushNotificationSender,
    PushNotificationConfigStore,
    PushNotificationEvent,
)
from a2a.types import TaskPushNotificationConfig
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict
from prometheus_client import Counter
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

from agentmesh.observability.logging import get_logger

log = get_logger(__name__)

PUSH_DELIVERIES = Counter(
    "agentmesh_push_deliveries_total", "Push notification deliveries", ["outcome"]
)

SIGNATURE_HEADER = "X-AgentMesh-Signature"
TIMESTAMP_HEADER = "X-AgentMesh-Timestamp"


def sign_payload(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify_push_signature(
    secret: str,
    *,
    body: bytes,
    timestamp: str,
    signature: str,
    tolerance_seconds: int = 300,
    now: float | None = None,
) -> bool:
    """Receiver-side check: valid HMAC and a timestamp within ``tolerance_seconds``."""
    try:
        age = abs((now if now is not None else time.time()) - float(timestamp))
    except ValueError:
        return False
    if age > tolerance_seconds:
        return False
    return hmac.compare_digest(sign_payload(secret, timestamp, body), signature)


@dataclass(frozen=True, slots=True)
class DeadLetter:
    task_id: str
    url: str
    error: str
    at: float


class DeadLetterQueue:
    """Bounded in-memory record of deliveries that exhausted their retries."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._items: deque[DeadLetter] = deque(maxlen=maxlen)

    def add(self, item: DeadLetter) -> None:
        self._items.append(item)

    def items(self) -> list[DeadLetter]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class SignedPushNotificationSender(BasePushNotificationSender):
    """Delivers task events to webhooks with retries and optional HMAC signing."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        config_store: PushNotificationConfigStore,
        *,
        signing_secret: str | None = None,
        push_url_validator: Callable[[str], Awaitable[bool]] | None = None,
        max_attempts: int = 3,
        dead_letters: DeadLetterQueue | None = None,
    ) -> None:
        super().__init__(httpx_client, config_store, push_url_validator=push_url_validator)
        self._secret = signing_secret
        self._max_attempts = max_attempts
        self.dead_letters = dead_letters or DeadLetterQueue()
        self._tails: dict[tuple[str, str], asyncio.Task[bool]] = {}
        self._pending: set[asyncio.Task[bool]] = set()

    async def send_notification(self, task_id: str, event: PushNotificationEvent) -> None:
        """Queue delivery in the background so webhook latency never blocks the RPC.

        Deliveries for the same (task, webhook) are chained to preserve event order.
        """
        for info in await self._config_store.get_info_for_dispatch(task_id):
            key = (task_id, info.url)
            previous = self._tails.get(key)
            delivery = asyncio.create_task(self._after(previous, event, info, task_id))
            self._tails[key] = delivery
            self._pending.add(delivery)
            delivery.add_done_callback(functools.partial(self._finished, key))

    async def _after(
        self,
        previous: asyncio.Task[bool] | None,
        event: PushNotificationEvent,
        info: TaskPushNotificationConfig,
        task_id: str,
    ) -> bool:
        if previous is not None:
            await asyncio.wait([previous])
        return await self._dispatch_notification(event, info, task_id)

    def _finished(self, key: tuple[str, str], task: asyncio.Task[bool]) -> None:
        self._pending.discard(task)
        if self._tails.get(key) is task:
            del self._tails[key]

    async def drain(self, timeout: float = 15.0) -> None:  # noqa: ASYNC109
        """Wait for in-flight deliveries (called on shutdown)."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout)

    async def _dispatch_notification(
        self, event: PushNotificationEvent, push_info: TaskPushNotificationConfig, task_id: str
    ) -> bool:
        url = push_info.url
        if self._push_url_validator is not None and not await self._push_url_validator(url):
            log.warning("push_url_rejected", task_id=task_id, url=url)
            PUSH_DELIVERIES.labels("rejected").inc()
            return False

        body = json.dumps(MessageToDict(to_stream_response(event)), separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if push_info.token:
            headers["X-A2A-Notification-Token"] = push_info.token
        auth = push_info.authentication
        if push_info.HasField("authentication") and auth.scheme and auth.credentials:
            headers["Authorization"] = f"{auth.scheme} {auth.credentials}"

        last_error = ""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    # Re-sign each attempt so the timestamp stays inside the receiver's window.
                    request_headers = dict(headers)
                    if self._secret:
                        ts = str(int(time.time()))
                        request_headers[TIMESTAMP_HEADER] = ts
                        request_headers[SIGNATURE_HEADER] = sign_payload(self._secret, ts, body)
                    response = await self._client.post(url, content=body, headers=request_headers)
                    response.raise_for_status()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.error("push_delivery_failed", task_id=task_id, url=url, error=last_error)
            self.dead_letters.add(DeadLetter(task_id, url, last_error, time.time()))
            PUSH_DELIVERIES.labels("dead_lettered").inc()
            return False
        PUSH_DELIVERIES.labels("delivered").inc()
        return True
