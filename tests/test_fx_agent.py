from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
from a2a.types import (
    GetTaskRequest,
    ListTasksRequest,
    TaskState,
)

from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, a2a_client, serve
from tests.helpers import context_id_of, final_state, task_id_of, user_request

RATES = StaticRates({"EUR": Decimal("0.9"), "GBP": Decimal("0.8"), "INR": Decimal("83")})


@pytest.fixture
def app(settings):
    return create_agent_app(
        FX_SPEC, build_fx_executor(settings, rates=RATES), settings, public_url=BASE_URL
    )


async def test_agent_card_is_published(app):
    async with serve(app) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
    assert card["name"] == "FX Conversion Agent"
    assert card["capabilities"]["streaming"] is True
    bindings = {i["protocolBinding"] for i in card["supportedInterfaces"]}
    assert bindings == {"JSONRPC", "HTTP+JSON"}
    assert card["skills"][0]["id"] == "currency-conversion"


@pytest.mark.parametrize("binding", ["JSONRPC", "HTTP+JSON"])
@pytest.mark.parametrize("streaming", [True, False])
async def test_conversion_over_every_transport(app, binding, streaming):
    async with serve(app) as http:
        client = await a2a_client(http, binding=binding, streaming=streaming)
        events = [e async for e in client.send_message(user_request("Convert 100 USD to EUR"))]
    assert final_state(events) == "TASK_STATE_COMPLETED"
    text = " ".join(str(e) for e in events)
    assert "90.00" in text


async def test_multi_turn_input_required_then_complete(app):
    async with serve(app) as http:
        client = await a2a_client(http, streaming=False)
        first = [e async for e in client.send_message(user_request("convert 250 to EUR"))]
        assert final_state(first) == "TASK_STATE_INPUT_REQUIRED"
        assert "converting from" in str(first[-1]).lower()

        second = [
            e
            async for e in client.send_message(
                user_request("USD", task_id=task_id_of(first), context_id=context_id_of(first))
            )
        ]
    assert final_state(second) == "TASK_STATE_COMPLETED"
    assert "225.00" in str(second[-1])


async def test_get_and_list_tasks(app):
    async with serve(app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(user_request("Convert 10 EUR to GBP"))]
        task_id = task_id_of(events)
        task = await client.get_task(GetTaskRequest(id=task_id))
        listing = await client.list_tasks(ListTasksRequest())
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task_id in {t.id for t in listing.tasks}


async def test_unsupported_pair_fails_cleanly(app):
    async with serve(app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(user_request("Convert 5 USD to JPY"))]
    assert final_state(events) == "TASK_STATE_FAILED"


async def test_ops_endpoints(app):
    async with serve(app) as http:
        assert (await http.get("/healthz")).json()["status"] == "ok"
        assert (await http.get("/readyz")).status_code == 200
        metrics = await http.get("/metrics")
    assert metrics.status_code == 200
    assert "agentmesh_http_requests_total" in metrics.text


async def test_request_id_is_propagated(app):
    async with serve(app) as http:
        response = await http.get("/healthz", headers={"x-request-id": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"


async def test_not_ready_before_lifespan(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as http:
        assert (await http.get("/readyz")).status_code == 503
