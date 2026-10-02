"""Expose an MCP server's tools as A2A skills."""

from __future__ import annotations

import json
from typing import Any

from a2a.helpers import get_data_parts, get_text_parts, new_data_part, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill

from agentmesh import __version__
from agentmesh.agents.mcp_bridge.connection import McpConnection, McpError, McpTool
from agentmesh.runtime import AgentSpec, MeshAgentExecutor, TaskInputError


def build_mcp_spec(tools: list[McpTool], *, slug: str = "mcp-bridge") -> AgentSpec:
    """One A2A skill per MCP tool; the card reflects the server's live tool list."""
    skills = [
        AgentSkill(
            id=tool.name,
            name=tool.name.replace("_", " ").replace("-", " ").title(),
            description=tool.description or f"MCP tool {tool.name}",
            tags=["mcp", "tool"],
            examples=[json.dumps({"tool": tool.name, "arguments": _example_arguments(tool)})],
        )
        for tool in tools
    ]
    return AgentSpec(
        slug=slug,
        name="MCP Tool Bridge",
        description="Exposes the tools of a Model Context Protocol server as A2A skills.",
        version=__version__,
        framework="model-context-protocol",
        streaming=False,
        input_modes=["application/json", "text/plain"],
        output_modes=["application/json", "text/plain"],
        skills=skills,
    )


def _example_arguments(tool: McpTool) -> dict[str, Any]:
    samples = {"string": "text", "integer": 1, "number": 1.0, "boolean": True}
    properties = tool.input_schema.get("properties", {})
    return {k: samples.get(v.get("type", "string")) for k, v in properties.items()}


def parse_tool_call(message_parts: list[Any], known: set[str]) -> tuple[str, dict[str, Any]]:
    """Accept ``{"tool", "arguments"}`` data, or text of the form ``tool {"json": "args"}``."""
    for data in get_data_parts(message_parts):
        if isinstance(data, dict) and "tool" in data:
            name, arguments = str(data["tool"]), data.get("arguments") or {}
            break
    else:
        text = " ".join(get_text_parts(message_parts)).strip()
        name, _, rest = text.partition(" ")
        try:
            arguments = json.loads(rest) if rest.strip() else {}
        except ValueError as exc:
            raise TaskInputError('Arguments must be a JSON object, e.g. add {"a": 1}') from exc
    if name not in known:
        raise TaskInputError(f"Unknown tool {name!r}. Available: {', '.join(sorted(known))}.")
    if not isinstance(arguments, dict):
        raise TaskInputError("Tool arguments must be a JSON object.")
    return name, arguments


class McpBridgeExecutor(MeshAgentExecutor):
    agent_name = "mcp-bridge"

    def __init__(
        self, connection: McpConnection, tools: list[McpTool], *, call_timeout: float = 30.0
    ):
        super().__init__()
        self._connection = connection
        self._tool_names = {t.name for t in tools}
        self._timeout = call_timeout

    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        message = context.message
        assert message is not None
        name, arguments = parse_tool_call(list(message.parts), self._tool_names)
        await updater.start_work(self.text(updater, f"Calling {name}"))
        try:
            result = await self._connection.call(name, arguments, timeout=self._timeout)
        except McpError as exc:
            await updater.failed(self.text(updater, str(exc)))
            return
        parts = [new_text_part(result.text)] if result.text else []
        if result.structured is not None:
            parts.append(new_data_part(result.structured))
        if parts:
            await updater.add_artifact(parts, name=name)
        message_out = self.text(updater, result.text or f"{name} finished")
        if result.is_error:
            await updater.failed(message_out)
        else:
            await updater.complete(message_out)

    async def aclose(self) -> None:
        await self._connection.aclose()
