"""Long-lived MCP client session, owned by a single task.

anyio cancel scopes must be entered and exited by the same task, so the transport and session
live inside one supervisor task. Request handlers talk to the session through it freely.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class McpError(RuntimeError):
    """The MCP server could not be reached or rejected the call."""


@dataclass(frozen=True, slots=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class McpResult:
    text: str
    structured: Any | None
    is_error: bool


class McpConnection:
    def __init__(
        self,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if bool(command) == bool(url):
            raise ValueError("configure exactly one of an MCP server command or URL")
        self._command = command
        self._url = url
        self._headers = headers
        self._env = env
        self._session: ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._lock = asyncio.Lock()

    async def _supervise(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                if self._command:
                    params = StdioServerParameters(
                        command=self._command[0], args=self._command[1:], env=self._env
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                else:
                    assert self._url is not None
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(self._url, headers=self._headers)
                    )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._session = None
            self._ready.set()

    async def open(self) -> None:
        async with self._lock:
            if self._task is not None:
                return
            self._task = asyncio.create_task(self._supervise())
            await self._ready.wait()
            if self._session is None:
                task, self._task = self._task, None
                with contextlib.suppress(BaseException):
                    await task
                raise McpError(f"could not connect to MCP server: {self._error!r}")

    async def _require(self) -> ClientSession:
        if self._session is None:
            await self.open()
        if self._session is None:
            raise McpError("MCP session is closed")
        return self._session

    async def list_tools(self) -> list[McpTool]:
        session = await self._require()
        tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(
                McpTool(t.name, t.description or "", dict(t.inputSchema)) for t in page.tools
            )
            cursor = page.nextCursor
            if not cursor:
                return tools

    async def call(
        self, name: str, arguments: dict[str, Any], *, timeout: float = 30.0
    ) -> McpResult:
        session = await self._require()
        try:
            async with asyncio.timeout(timeout):
                result = await session.call_tool(name, arguments)
        except TimeoutError as exc:
            raise McpError(f"tool {name!r} timed out after {timeout:.0f}s") from exc
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        return McpResult(text, getattr(result, "structuredContent", None), bool(result.isError))

    async def aclose(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        with contextlib.suppress(BaseException):
            await self._task
        self._task = None
