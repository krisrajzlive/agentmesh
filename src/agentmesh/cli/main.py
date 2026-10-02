"""``agentmesh`` command-line interface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Annotated, Any

import typer
import uvicorn
from a2a.types import TaskState
from google.protobuf.json_format import MessageToDict
from rich.console import Console
from rich.table import Table

from agentmesh.agents.catalog import UnknownAgentError, available_agents, load_agent
from agentmesh.client import AgentClient, TaskResult, apply_event
from agentmesh.config import Settings, get_settings
from agentmesh.observability import configure_logging
from agentmesh.registry import RegistryClient, RegistryClientError
from agentmesh.runtime import create_agent_app
from agentmesh.security import issue_token

console = Console()
err = Console(stderr=True)

app = typer.Typer(no_args_is_help=True, help="AgentMesh: A2A multi-agent platform.")
agent_app = typer.Typer(no_args_is_help=True, help="Talk to a running A2A agent.")
registry_app = typer.Typer(no_args_is_help=True, help="Manage the agent registry.")
app.add_typer(agent_app, name="agent")
app.add_typer(registry_app, name="registry")

ApiKey = Annotated[
    str | None, typer.Option("--api-key", envvar="AGENTMESH_API_KEY", help="API key to send.")
]
Bearer = Annotated[
    str | None, typer.Option("--token", envvar="AGENTMESH_TOKEN", help="Bearer token to send.")
]


def _run[T](coro: Awaitable[T]) -> T:
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except RegistryClientError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        err.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    return settings


# ------------------------------------------------------------------ servers


@app.command("agents")
def list_agents() -> None:
    """List the agents this installation can host."""
    for slug in available_agents():
        typer.echo(slug)


@app.command()
def serve(
    agent: Annotated[str, typer.Argument(help="Agent slug (see `agentmesh agents`).")],
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 9000,
    public_url: Annotated[
        str | None, typer.Option(help="URL peers use to reach this agent.")
    ] = None,
) -> None:
    """Run one agent as an A2A server."""
    settings = _settings()
    try:
        spec, executor = load_agent(agent, settings)
    except UnknownAgentError as exc:
        raise typer.BadParameter(str(exc)) from exc
    asgi = create_agent_app(
        spec, executor, settings, public_url=public_url or f"http://localhost:{port}"
    )
    uvicorn.run(asgi, host=host, port=port, log_config=None, access_log=False)


@app.command("run-registry")
def run_registry(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 9100,
) -> None:
    """Run the agent registry service."""
    from agentmesh.registry import create_registry_app

    settings = _settings()
    uvicorn.run(
        create_registry_app(settings), host=host, port=port, log_config=None, access_log=False
    )


@app.command("run-gateway")
def run_gateway(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 9200,
    public_url: Annotated[str | None, typer.Option(help="Externally visible gateway URL.")] = None,
) -> None:
    """Run the API gateway (needs AGENTMESH_REGISTRY_URL or AGENTMESH_STATIC_AGENT_URLS)."""
    from agentmesh.gateway import create_gateway_app
    from agentmesh.orchestrator import RegistryDirectory, StaticDirectory

    settings = _settings()
    if settings.registry_url:
        key = settings.outbound_api_key.get_secret_value() if settings.outbound_api_key else None
        directory: Any = RegistryDirectory(RegistryClient(settings.registry_url, api_key=key))
    elif settings.static_agent_urls:
        directory = StaticDirectory(settings.static_agent_urls)
    else:
        raise typer.BadParameter("set AGENTMESH_REGISTRY_URL or AGENTMESH_STATIC_AGENT_URLS")
    url = public_url or settings.gateway_public_url or f"http://localhost:{port}"
    asgi = create_gateway_app(settings, directory, public_url=url)
    uvicorn.run(asgi, host=host, port=port, log_config=None, access_log=False)


# ------------------------------------------------------------------- tokens


@app.command("token")
def token(
    subject: Annotated[str, typer.Argument(help="Principal name (becomes the task owner).")],
    ttl: Annotated[int, typer.Option(help="Lifetime in seconds.")] = 3600,
    scope: Annotated[list[str] | None, typer.Option(help="Scope to include (repeatable).")] = None,
) -> None:
    """Mint a JWT signed with AGENTMESH_JWT_SECRET."""
    settings = get_settings()
    if settings.jwt_secret is None:
        raise typer.BadParameter("AGENTMESH_JWT_SECRET is not set")
    typer.echo(
        issue_token(
            settings.jwt_secret.get_secret_value(),
            subject,
            audience=settings.jwt_audience,
            ttl_seconds=ttl,
            scopes=scope or [],
            algorithm=settings.jwt_algorithm,
            issuer=settings.jwt_issuer,
        )
    )


# ------------------------------------------------------------- agent client


def _print_result(result: TaskResult) -> None:
    console.print(f"[bold]task[/bold] {result.task_id}  [bold]state[/bold] {result.state_name}")
    if result.output_text:
        console.print(result.output_text)
    for artifact in result.artifacts:
        console.print(f"[dim]artifact[/dim] {artifact.name or artifact.artifact_id}")
        for item in artifact.data:
            console.print_json(json.dumps(item))


@agent_app.command("card")
def agent_card(url: str, api_key: ApiKey = None, token: Bearer = None) -> None:
    """Fetch and print an agent's card."""

    async def go() -> None:
        async with await AgentClient.connect(url, api_key=api_key, bearer_token=token) as client:
            console.print_json(json.dumps(MessageToDict(client.card)))

    _run(go())


@agent_app.command("send")
def agent_send(
    url: str,
    message: str,
    stream: Annotated[bool, typer.Option(help="Print updates as they arrive.")] = False,
    task_id: Annotated[str | None, typer.Option(help="Continue an existing task.")] = None,
    context_id: Annotated[str | None, typer.Option(help="Continue a conversation.")] = None,
    api_key: ApiKey = None,
    token: Bearer = None,
) -> None:
    """Send a message and show the result."""

    async def go() -> None:
        async with await AgentClient.connect(
            url, api_key=api_key, bearer_token=token, streaming=stream
        ) as client:
            result = TaskResult()
            async for event in client.stream(message, task_id=task_id, context_id=context_id):
                apply_event(result, event)
                if stream:
                    console.print(f"[dim]{result.state_name}[/dim] {result.text}")
            _print_result(result)
            if result.needs_input:
                console.print(
                    f"[yellow]input required[/yellow] — reply with "
                    f"--task-id {result.task_id} --context-id {result.context_id}"
                )

    _run(go())


@agent_app.command("tasks")
def agent_tasks(url: str, api_key: ApiKey = None, token: Bearer = None) -> None:
    """List tasks."""

    async def go() -> None:
        async with await AgentClient.connect(url, api_key=api_key, bearer_token=token) as client:
            table = Table()
            table.add_column("task", no_wrap=True)
            table.add_column("state")
            table.add_column("context", overflow="fold")
            for task in await client.list_tasks():
                state = TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
                table.add_row(task.id, state, task.context_id)
            console.print(table)

    _run(go())


@agent_app.command("cancel")
def agent_cancel(url: str, task_id: str, api_key: ApiKey = None, token: Bearer = None) -> None:
    """Cancel a running task."""

    async def go() -> None:
        async with await AgentClient.connect(url, api_key=api_key, bearer_token=token) as client:
            task = await client.cancel(task_id)
            console.print(f"{task.id}: {task.status.state}")

    _run(go())


# ----------------------------------------------------------------- registry


def _registry(url: str, api_key: str | None, token: str | None) -> RegistryClient:
    return RegistryClient(url, api_key=api_key, bearer_token=token)


RegistryUrl = Annotated[
    str, typer.Option("--registry", envvar="AGENTMESH_REGISTRY_URL", help="Registry base URL.")
]


@registry_app.command("register")
def registry_register(
    agent_url: str, registry: RegistryUrl, api_key: ApiKey = None, token: Bearer = None
) -> None:
    """Register an agent by its base URL."""

    async def go() -> None:
        client = _registry(registry, api_key, token)
        try:
            record = await client.register(agent_url)
        finally:
            await client.close()
        console.print(f"registered [bold]{record.id}[/bold] ({record.framework}) at {record.url}")

    _run(go())


@registry_app.command("list")
def registry_list(
    registry: RegistryUrl,
    skill: str | None = None,
    tag: str | None = None,
    query: Annotated[str | None, typer.Option("--query", "-q")] = None,
    api_key: ApiKey = None,
    token: Bearer = None,
) -> None:
    """List registered agents."""

    async def go() -> None:
        client = _registry(registry, api_key, token)
        try:
            records = await client.list(skill=skill, tag=tag, query=query)
        finally:
            await client.close()
        table = Table(show_lines=False)
        table.add_column("id", no_wrap=True)
        for column in ("framework", "status", "skills", "url"):
            table.add_column(column, overflow="fold")
        for r in records:
            table.add_row(
                r.id, r.framework, r.status.value, ", ".join(s.id for s in r.skills), r.url
            )
        console.print(table)

    _run(go())


@registry_app.command("remove")
def registry_remove(
    agent_id: str, registry: RegistryUrl, api_key: ApiKey = None, token: Bearer = None
) -> None:
    """Remove an agent from the registry."""

    async def go() -> None:
        client = _registry(registry, api_key, token)
        try:
            await client.deregister(agent_id)
        finally:
            await client.close()
        console.print(f"removed {agent_id}")

    _run(go())


# ------------------------------------------------------------------ webhooks


@app.command("webhook-listen")
def webhook_listen(
    port: int = 9300,
    host: str = "127.0.0.1",
    secret: Annotated[
        str | None,
        typer.Option(envvar="AGENTMESH_PUSH_SIGNING_SECRET", help="Verify HMAC signatures."),
    ] = None,
) -> None:
    """Run a webhook receiver that verifies and prints A2A push notifications."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from agentmesh.runtime.push import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify_push_signature

    async def receive(request: Request) -> JSONResponse:
        body = await request.body()
        if secret and not verify_push_signature(
            secret,
            body=body,
            timestamp=request.headers.get(TIMESTAMP_HEADER, ""),
            signature=request.headers.get(SIGNATURE_HEADER, ""),
        ):
            console.print("[red]rejected: bad signature[/red]")
            return JSONResponse({"error": "invalid signature"}, status_code=401)
        console.print_json(body.decode())
        return JSONResponse({"ok": True})

    uvicorn.run(
        Starlette(routes=[Route("/", receive, methods=["POST"])]),
        host=host,
        port=port,
        log_level="warning",
    )
