"""End-to-end: real uvicorn servers on real sockets, driven through the CLI."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.cli.main import app
from agentmesh.config import Settings
from agentmesh.registry import create_registry_app
from agentmesh.runtime import create_agent_app

runner = CliRunner()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def run_server(asgi) -> Iterator[str]:
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(asgi, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/healthz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        raise RuntimeError("server did not start")
    try:
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


@pytest.fixture
def batch_agent():
    config = settings()
    port_holder = {}

    # The card advertises its own public URL, so build the app after picking the port.
    port = free_port()
    port_holder["url"] = f"http://127.0.0.1:{port}"
    asgi = create_agent_app(BATCH_SPEC, BatchExecutor(), config, public_url=port_holder["url"])
    server = uvicorn.Server(uvicorn.Config(asgi, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{port_holder['url']}/readyz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield port_holder["url"]
    server.should_exit = True
    thread.join(timeout=10)


def test_cli_agents_and_token(monkeypatch):
    listed = runner.invoke(app, ["agents"])
    assert listed.exit_code == 0
    assert {"fx", "orchestrator", "analytics", "quant", "mcp-bridge"} <= set(listed.output.split())

    monkeypatch.setenv("AGENTMESH_JWT_SECRET", "s" * 40)
    from agentmesh.config import get_settings

    get_settings.cache_clear()
    minted = runner.invoke(app, ["token", "ci-bot", "--ttl", "60", "--scope", "tasks:write"])
    get_settings.cache_clear()
    assert minted.exit_code == 0
    import jwt

    claims = jwt.decode(minted.output.strip(), "s" * 40, algorithms=["HS256"], audience="agentmesh")
    assert claims["sub"] == "ci-bot"
    assert claims["scope"] == "tasks:write"


def test_cli_talks_to_a_real_agent_over_real_http(batch_agent):
    card = runner.invoke(app, ["agent", "card", batch_agent])
    assert card.exit_code == 0, card.output
    assert json.loads(card.output)["name"] == "Batch Processing Agent"

    sent = runner.invoke(app, ["agent", "send", batch_agent, "hash\nalpha\nbeta", "--stream"])
    assert sent.exit_code == 0, sent.output
    assert "completed" in sent.output
    assert "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8" in sent.output

    tasks = runner.invoke(app, ["agent", "tasks", batch_agent])
    assert tasks.exit_code == 0
    assert "completed" in tasks.output


def test_cli_reports_errors_cleanly():
    result = runner.invoke(app, ["agent", "card", "http://127.0.0.1:9"])
    assert result.exit_code == 1


def test_cli_registry_commands_against_a_live_registry(batch_agent):
    config = settings(registry_allow_private_urls=True)
    with run_server(create_registry_app(config)) as registry:
        registered = runner.invoke(
            app, ["registry", "register", batch_agent, "--registry", registry]
        )
        assert registered.exit_code == 0, registered.output
        assert "batch-processing-agent" in registered.output

        listed = runner.invoke(
            app, ["registry", "list", "--registry", registry, "--skill", "batch-hash"]
        )
        assert listed.exit_code == 0
        assert "batch-processing" in listed.output

        removed = runner.invoke(
            app, ["registry", "remove", "batch-processing-agent", "--registry", registry]
        )
        assert removed.exit_code == 0
        empty = runner.invoke(app, ["registry", "list", "--registry", registry])
        assert "batch-processing" not in empty.output
