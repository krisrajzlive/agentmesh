"""MCP tool bridge (requires the ``mcp`` extra)."""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentmesh.agents.endpoint import AgentConfigurationError
from agentmesh.agents.mcp_bridge.connection import McpConnection, McpError, McpResult, McpTool
from agentmesh.agents.mcp_bridge.executor import McpBridgeExecutor, build_mcp_spec, parse_tool_call
from agentmesh.config import Settings
from agentmesh.runtime.spec import AgentSpec

__all__ = [
    "McpBridgeExecutor",
    "McpConnection",
    "McpError",
    "McpResult",
    "McpTool",
    "build_agent",
    "build_mcp_spec",
    "parse_tool_call",
]


def split_command(command: str) -> list[str]:
    """Split a command line; on Windows backslashes are paths, so quotes are stripped by hand."""
    if sys.platform != "win32":
        return shlex.split(command)
    return [token.strip('"') for token in shlex.split(command, posix=False)]


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion from sync code, even when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def build_agent(settings: Settings) -> tuple[AgentSpec, McpBridgeExecutor]:
    """Discover the server's tools once at start-up so the Agent Card lists them."""
    if not (settings.mcp_server_command or settings.mcp_server_url):
        raise AgentConfigurationError(
            "set AGENTMESH_MCP_SERVER_COMMAND or AGENTMESH_MCP_SERVER_URL to bridge an MCP server"
        )
    command = split_command(settings.mcp_server_command) if settings.mcp_server_command else None

    async def discover() -> list[McpTool]:
        probe = McpConnection(command=command, url=settings.mcp_server_url)
        try:
            return await probe.list_tools()
        finally:
            await probe.aclose()

    try:
        tools = _run_sync(discover())
    except McpError as exc:
        raise AgentConfigurationError(str(exc)) from exc
    allowed = set(settings.mcp_tool_allowlist)
    if allowed:
        tools = [t for t in tools if t.name in allowed]
    if not tools:
        raise AgentConfigurationError("the MCP server exposes no (allowed) tools")
    connection = McpConnection(command=command, url=settings.mcp_server_url)
    return build_mcp_spec(tools), McpBridgeExecutor(connection, tools)
