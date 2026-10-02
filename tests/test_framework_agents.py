"""Google ADK and Microsoft Agent Framework agents served over A2A, driven by scripted models."""

from __future__ import annotations

import pytest

from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, a2a_client, serve
from tests.helpers import final_state, user_request
from tests.support import ScriptedAdkModel, scripted_maf_client

pytest.importorskip("google.adk")
pytest.importorskip("agent_framework")


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


async def test_agent_framework_agent_runs_tools_and_answers_over_a2a(settings):
    app = create_agent_app(
        QUANT_SPEC, build_quant_executor(scripted_maf_client()), settings, public_url=BASE_URL
    )
    async with serve(app) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
        a2a = await a2a_client(http, streaming=False)
        events = [
            e async for e in a2a.send_message(user_request("Payment on 250k at 6.5% for 30y?"))
        ]
    assert card["name"] == "Quantitative Analysis Agent"
    assert final_state(events) == "TASK_STATE_COMPLETED"
    assert "1580.17" in " ".join(str(e) for e in events)
