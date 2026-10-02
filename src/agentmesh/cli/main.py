"""``agentmesh`` command-line interface."""

from __future__ import annotations

import typer
import uvicorn

from agentmesh.agents.catalog import UnknownAgentError, available_agents, load_agent
from agentmesh.config import get_settings
from agentmesh.observability import configure_logging
from agentmesh.runtime import create_agent_app

app = typer.Typer(no_args_is_help=True, help="AgentMesh: A2A multi-agent platform.")


@app.command("agents")
def list_agents() -> None:
    """List the agents this installation can host."""
    for slug in available_agents():
        typer.echo(slug)


@app.command()
def serve(
    agent: str = typer.Argument(..., help="Agent slug (see `agentmesh agents`)."),
    host: str = typer.Option("0.0.0.0", help="Bind address."),  # noqa: S104
    port: int = typer.Option(9000, help="Bind port."),
    public_url: str = typer.Option(
        None, help="URL other agents use to reach this one (defaults to http://localhost:<port>)."
    ),
) -> None:
    """Run one agent as an A2A server."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    try:
        spec, executor = load_agent(agent, settings)
    except UnknownAgentError as exc:
        raise typer.BadParameter(str(exc)) from exc
    asgi = create_agent_app(
        spec, executor, settings, public_url=public_url or f"http://localhost:{port}"
    )
    uvicorn.run(asgi, host=host, port=port, log_config=None, access_log=False)
