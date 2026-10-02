"""High-level async client for A2A agents (ours or any spec-compliant server)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

import httpx
from a2a.client import ClientConfig, create_client
from a2a.client.client import Client
from a2a.helpers import new_message, new_text_part
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetExtendedAgentCardRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTasksRequest,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskPushNotificationConfig,
    TaskState,
)

from agentmesh.client.results import TaskResult, apply_event


class AgentClient:
    """Thin, typed facade over the SDK client with auth and result folding."""

    def __init__(
        self, client: Client, card: AgentCard, http: httpx.AsyncClient, *, owns_http: bool
    ):
        self._client = client
        self.card = card
        self._http = http
        self._owns_http = owns_http

    @classmethod
    async def connect(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        http: httpx.AsyncClient | None = None,
        streaming: bool = True,
        binding: str = "JSONRPC",
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> Self:
        """Discover the agent at ``url`` (via its Agent Card) and return a connected client."""
        owns_http = http is None
        http = http or httpx.AsyncClient(timeout=timeout)
        if api_key:
            http.headers["x-api-key"] = api_key
        if bearer_token:
            http.headers["authorization"] = f"Bearer {bearer_token}"
        if headers:
            http.headers.update(headers)
        config = ClientConfig(
            httpx_client=http, streaming=streaming, supported_protocol_bindings=[binding]
        )
        try:
            client = await create_client(url, client_config=config)
        except BaseException:
            if owns_http:
                await http.aclose()
            raise
        card = await _fetch_card(http, url)
        return cls(client, card, http, owns_http=owns_http)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        # The SDK client closes the HTTP client it was given, so a caller-supplied
        # (shared) client must be left alone.
        if self._owns_http:
            await self._client.close()
            await self._http.aclose()

    # ------------------------------------------------------------------ messaging

    @staticmethod
    def _request(
        content: str | Sequence[Part],
        *,
        task_id: str | None,
        context_id: str | None,
        push: TaskPushNotificationConfig | None,
        return_immediately: bool,
    ) -> SendMessageRequest:
        parts = [new_text_part(content)] if isinstance(content, str) else list(content)
        message = new_message(parts, context_id=context_id, task_id=task_id, role=Role.ROLE_USER)
        request = SendMessageRequest(message=message)
        configuration = SendMessageConfiguration(return_immediately=return_immediately)
        if push is not None:
            configuration.task_push_notification_config.CopyFrom(push)
        request.configuration.CopyFrom(configuration)
        return request

    async def stream(
        self,
        content: str | Sequence[Part],
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        push: TaskPushNotificationConfig | None = None,
        return_immediately: bool = False,
    ) -> AsyncIterator[StreamResponse]:
        """Yield raw stream events as the agent produces them."""
        request = self._request(
            content,
            task_id=task_id,
            context_id=context_id,
            push=push,
            return_immediately=return_immediately,
        )
        async for event in self._client.send_message(request):
            yield event

    async def send(
        self,
        content: str | Sequence[Part],
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        push: TaskPushNotificationConfig | None = None,
        return_immediately: bool = False,
    ) -> TaskResult:
        """Send a message and fold the response stream into a :class:`TaskResult`."""
        result = TaskResult()
        async for event in self.stream(
            content,
            task_id=task_id,
            context_id=context_id,
            push=push,
            return_immediately=return_immediately,
        ):
            apply_event(result, event)
        return result

    # --------------------------------------------------------------------- tasks

    async def get_task(self, task_id: str, *, history_length: int | None = None) -> Task:
        request = GetTaskRequest(id=task_id)
        if history_length is not None:
            request.history_length = history_length
        return await self._client.get_task(request)

    async def list_tasks(
        self,
        *,
        context_id: str | None = None,
        state: TaskState | None = None,
        page_size: int = 50,
        page_token: str = "",
    ) -> list[Task]:
        request = ListTasksRequest(page_size=page_size, page_token=page_token)
        if context_id:
            request.context_id = context_id
        if state is not None:
            request.status = state
        return list((await self._client.list_tasks(request)).tasks)

    async def cancel(self, task_id: str) -> Task:
        return await self._client.cancel_task(CancelTaskRequest(id=task_id))

    async def subscribe(self, task_id: str) -> AsyncIterator[StreamResponse]:
        async for event in self._client.subscribe(SubscribeToTaskRequest(id=task_id)):
            yield event

    # ------------------------------------------------------------ push notifications

    async def set_push_config(
        self, task_id: str, url: str, *, token: str = "", config_id: str = ""
    ) -> TaskPushNotificationConfig:
        config = TaskPushNotificationConfig(task_id=task_id, url=url, token=token, id=config_id)
        return await self._client.create_task_push_notification_config(config)

    async def list_push_configs(self, task_id: str) -> list[TaskPushNotificationConfig]:
        response = await self._client.list_task_push_notification_configs(
            ListTaskPushNotificationConfigsRequest(task_id=task_id)
        )
        return list(response.configs)

    async def delete_push_config(self, task_id: str, config_id: str) -> None:
        await self._client.delete_task_push_notification_config(
            DeleteTaskPushNotificationConfigRequest(task_id=task_id, id=config_id)
        )

    async def extended_card(self) -> AgentCard:
        return await self._client.get_extended_agent_card(GetExtendedAgentCardRequest())


async def _fetch_card(http: httpx.AsyncClient, url: str) -> AgentCard:
    from a2a.client import A2ACardResolver

    return await A2ACardResolver(http, url).get_agent_card()
