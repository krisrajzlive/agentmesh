from __future__ import annotations

import sys
from pathlib import Path

import pytest
from a2a.helpers import new_data_part, new_text_part

from agentmesh.agents.catalog import load_agent
from agentmesh.agents.endpoint import AgentConfigurationError
from agentmesh.client import AgentClient
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, serve

pytest.importorskip("mcp")

SERVER = Path(__file__).parent / "fixtures" / "calc_mcp_server.py"


def settings_for_server(**kwargs) -> Settings:
    command = f'"{sys.executable}" "{SERVER}"'
    return Settings(_env_file=None, mcp_server_command=command, **kwargs)  # type: ignore[call-arg]


def test_card_lists_one_skill_per_mcp_tool():
    spec, _ = load_agent("mcp-bridge", settings_for_server())
    assert {s.id for s in spec.skills} == {"add", "shout", "explode"}
    add = next(s for s in spec.skills if s.id == "add")
    assert add.description == "Add two integers."
    assert '"a"' in add.examples[0]


def test_allowlist_filters_tools():
    spec, _ = load_agent("mcp-bridge", settings_for_server(mcp_tool_allowlist="shout"))
    assert [s.id for s in spec.skills] == ["shout"]


def test_unconfigured_bridge_explains_what_is_missing():
    with pytest.raises(AgentConfigurationError, match="MCP_SERVER"):
        load_agent("mcp-bridge", Settings(_env_file=None))  # type: ignore[call-arg]


async def test_tools_are_callable_over_a2a():
    settings = settings_for_server()
    spec, executor = load_agent("mcp-bridge", settings)
    app = create_agent_app(spec, executor, settings, public_url=BASE_URL)
    async with serve(app) as http:
        client = await AgentClient.connect(BASE_URL, http=http, streaming=False)

        structured = await client.send(
            [new_data_part({"tool": "add", "arguments": {"a": 2, "b": 40}})]
        )
        assert structured.succeeded
        assert structured.text == "42"
        assert structured.artifact("add") is not None

        textual = await client.send([new_text_part('shout {"text": "hello mesh"}')])
        assert textual.succeeded
        assert textual.text == "HELLO MESH"

        unknown = await client.send("nope {}")
        assert unknown.state_name == "rejected"
        assert "Available" in unknown.text

        bad_json = await client.send("add {not json")
        assert bad_json.state_name == "rejected"

        broken = await client.send("explode")
        assert broken.state_name == "failed"
