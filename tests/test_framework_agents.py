"""Google ADK and Microsoft Agent Framework agents served over A2A, driven by scripted models."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, a2a_client, serve
from tests.helpers import final_state, user_request

pytest.importorskip("google.adk")
pytest.importorskip("agent_framework")

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types as genai
from openai import AsyncOpenAI

from agentmesh.agents.analytics import ANALYTICS_SPEC, build_analytics_executor
from agentmesh.agents.analytics.tools import (
    extract_keywords,
    redact_pii,
    text_statistics,
)
from agentmesh.agents.quant import QUANT_SPEC, build_quant_executor
from agentmesh.agents.quant.tools import (
    compound_growth,
    descriptive_statistics,
    loan_payment,
)

# --------------------------------------------------------------------------- tools


def test_text_tools():
    stats = text_statistics("One two three. Four five!")
    assert stats["words"] == 5 and stats["sentences"] == 2
    assert extract_keywords("cats chase mice. cats nap. mice hide.", 2)[0] == {
        "keyword": "cats",
        "count": 2,
    }
    redacted = redact_pii("mail jane@example.com or call +1 415 555 0100")
    assert "[EMAIL]" in redacted["redacted_text"]
    assert redacted["emails_masked"] == 1 and redacted["phones_masked"] == 1


def test_quant_tools():
    stats = descriptive_statistics.func([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == 2.5 and stats["median"] == 2.5
    loan = loan_payment.func(250_000, 6.5, 30)
    assert loan["monthly_payment"] == pytest.approx(1580.17, abs=0.01)
    assert compound_growth.func(1000, 10, 1, 1)["future_value"] == 1100.0
    with pytest.raises(ValueError, match="empty"):
        descriptive_statistics.func([])


# ----------------------------------------------------------------- Google ADK agent


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


async def test_adk_agent_runs_tools_and_answers_over_a2a(settings):
    app = create_agent_app(
        ANALYTICS_SPEC,
        build_analytics_executor(ScriptedAdkModel()),
        settings,
        public_url=BASE_URL,
    )
    async with serve(app) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(user_request("Count the words please"))]
    assert card["name"] == "Text Analytics Agent"
    assert final_state(events) == "TASK_STATE_COMPLETED"
    assert "4 words" in str(events[-1])


# ------------------------------------------------- Microsoft Agent Framework agent


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


def _scripted_openai(request: httpx.Request) -> httpx.Response:
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


async def test_agent_framework_agent_runs_tools_and_answers_over_a2a(settings):
    from agent_framework.openai import OpenAIChatCompletionClient

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_scripted_openai))
    client = OpenAIChatCompletionClient(
        model="scripted",
        async_client=AsyncOpenAI(
            api_key="test", base_url="http://llm.test/v1", http_client=http_client
        ),
    )
    app = create_agent_app(QUANT_SPEC, build_quant_executor(client), settings, public_url=BASE_URL)
    async with serve(app) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
        a2a = await a2a_client(http, streaming=False)
        events = [
            e async for e in a2a.send_message(user_request("Payment on 250k at 6.5% for 30y?"))
        ]
    assert card["name"] == "Quantitative Analysis Agent"
    assert final_state(events) == "TASK_STATE_COMPLETED"
    assert "1580.17" in " ".join(str(e) for e in events)
