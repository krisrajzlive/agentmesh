"""Shared test doubles: scripted models for the ADK and Agent Framework agents, and a mesh."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal

import httpx
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types as genai
from openai import AsyncOpenAI
from starlette.applications import Starlette

from agentmesh.agents.analytics import ANALYTICS_SPEC, build_analytics_executor
from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.documents import DOCUMENTS_SPEC, DocumentsExecutor
from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.agents.quant import QUANT_SPEC, build_quant_executor
from agentmesh.config import Settings
from agentmesh.orchestrator import ORCHESTRATOR_SPEC, RuleBasedPlanner, StaticDirectory
from agentmesh.orchestrator.executor import OrchestratorExecutor
from agentmesh.runtime import create_agent_app

# ----------------------------------------------------------------------- Google ADK


class ScriptedAdkModel(BaseLlm):
    """Calls ``text_statistics`` once, then summarises the tool result."""

    model: str = "scripted"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        last = llm_request.contents[-1]
        responses = [p.function_response for p in last.parts if p.function_response]
        if responses:
            words = responses[0].response.get("words", "?")
            text = f"The passage contains {words} words."
            yield LlmResponse(content=genai.Content(role="model", parts=[genai.Part(text=text)]))
            return
        call = genai.FunctionCall(name="text_statistics", args={"text": "alpha beta gamma delta"})
        yield LlmResponse(
            content=genai.Content(role="model", parts=[genai.Part(function_call=call)])
        )


# ------------------------------------------------------------- Microsoft Agent Framework


def _chunk(delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse(chunks: list[str]) -> httpx.Response:
    body = "".join(chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def scripted_openai(request: httpx.Request) -> httpx.Response:
    """Chat-completions server: first asks for the loan tool, then summarises its result."""
    body = json.loads(request.content)
    assert body.get("stream"), "the Agent Framework executor streams"
    tool_results = [m for m in body["messages"] if m["role"] == "tool"]
    if tool_results:
        payment = json.loads(tool_results[-1]["content"])
        text = f"Your monthly payment is {payment['monthly_payment']}."
        return _sse([_chunk({"role": "assistant", "content": text}), _chunk({}, "stop")])
    call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "loan_payment",
            "arguments": json.dumps({"principal": 250000, "annual_rate_percent": 6.5, "years": 30}),
        },
    }
    return _sse([_chunk({"role": "assistant", "tool_calls": [call]}), _chunk({}, "tool_calls")])


def scripted_maf_client():  # type: ignore[no-untyped-def]
    from agent_framework.openai import OpenAIChatCompletionClient

    http = httpx.AsyncClient(transport=httpx.MockTransport(scripted_openai))
    return OpenAIChatCompletionClient(
        model="scripted",
        async_client=AsyncOpenAI(api_key="test", base_url="http://llm.test/v1", http_client=http),
    )


# ------------------------------------------------------------------------------ mesh


class Switchboard(httpx.AsyncBaseTransport):
    """Routes requests to in-process ASGI apps by hostname; hosts in ``down`` refuse connections."""

    def __init__(self, apps: dict[str, Starlette]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}
        self.down: set[str] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in self.down or host not in self._transports:
            raise httpx.ConnectError("connection refused", request=request)
        return await self._transports[host].handle_async_request(request)


@dataclass
class Mesh:
    """Five agents (native, Google ADK, Microsoft Agent Framework) plus an orchestrator."""

    outbound: httpx.AsyncClient  # reaches every agent by hostname
    orchestrator: Starlette
    urls: dict[str, str]
    apps: dict[str, Starlette] = field(default_factory=dict)

    @asynccontextmanager
    async def frontdoor(self) -> AsyncIterator[httpx.AsyncClient]:
        """HTTP client bound to the orchestrator's ASGI app."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.orchestrator), base_url="http://orch.test"
        ) as client:
            yield client


@asynccontextmanager
async def build_mesh(
    settings: Settings | None = None, *, unreachable: tuple[str, ...] = ()
) -> AsyncIterator[Mesh]:
    """Start the mesh. Agents in ``unreachable`` are advertised but refuse connections."""
    settings = settings or Settings(_env_file=None)  # type: ignore[call-arg]
    specs = {
        "fx": (
            FX_SPEC,
            build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")})),
        ),
        "batch": (BATCH_SPEC, BatchExecutor()),
        "docs": (DOCUMENTS_SPEC, DocumentsExecutor()),
        "analytics": (ANALYTICS_SPEC, build_analytics_executor(ScriptedAdkModel())),
        "quant": (QUANT_SPEC, build_quant_executor(scripted_maf_client())),
    }
    urls = {name: f"http://{name}.test" for name in specs}
    apps = {
        name: create_agent_app(spec, executor, settings, public_url=urls[name])
        for name, (spec, executor) in specs.items()
    }
    async with AsyncExitStack() as stack:
        for app in apps.values():
            await stack.enter_async_context(app.router.lifespan_context(app))
        switchboard = Switchboard({f"{n}.test": a for n, a in apps.items()})
        outbound = httpx.AsyncClient(transport=switchboard)
        directory = StaticDirectory(list(urls.values()), http=outbound, ttl_seconds=3600)
        await directory.agents()  # cache the catalogue while everything is up
        switchboard.down.update(f"{n}.test" for n in unreachable)
        orchestrator_executor = OrchestratorExecutor(
            directory, RuleBasedPlanner(), settings, http=outbound
        )
        orchestrator = create_agent_app(
            ORCHESTRATOR_SPEC, orchestrator_executor, settings, public_url="http://orch.test"
        )
        await stack.enter_async_context(orchestrator.router.lifespan_context(orchestrator))
        yield Mesh(outbound, orchestrator, urls, apps)
        await outbound.aclose()
