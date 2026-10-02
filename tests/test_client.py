from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from a2a.helpers import new_data_part
from a2a.types import TaskState

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.client import AgentClient
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, serve


@pytest.fixture
def fx_app(settings):
    executor = build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")}))
    return create_agent_app(FX_SPEC, executor, settings, public_url=BASE_URL)


async def test_send_returns_folded_result(fx_app):
    async with serve(fx_app) as http:
        async with _connect(http) as c:
            assert c.card.name == "FX Conversion Agent"
            result = await c.send("Convert 100 USD to EUR")
    assert result.succeeded and result.state_name == "completed"
    assert "90.00 EUR" in result.text
    artifact = result.artifact("conversion")
    assert artifact is not None
    assert artifact.data[0]["converted"] == "90.00"


async def test_multi_turn_conversation_via_ids(fx_app):
    async with serve(fx_app) as http, _connect(http) as c:
        first = await c.send("convert 250 to EUR")
        assert first.needs_input and "from" in first.text.lower()
        second = await c.send("USD", task_id=first.task_id, context_id=first.context_id)
    assert second.succeeded and "225.00" in second.text


async def test_streaming_and_non_streaming_agree(fx_app):
    async with serve(fx_app) as http:
        async with _connect(http, streaming=True) as streaming:
            a = await streaming.send("Convert 10 USD to EUR")
        async with _connect(http, streaming=False) as polling:
            b = await polling.send("Convert 10 USD to EUR")
    assert a.text == b.text and a.state == b.state


async def test_task_queries_and_rest_binding(fx_app):
    async with serve(fx_app) as http, _connect(http, binding="HTTP+JSON") as c:
        result = await c.send("Convert 10 USD to EUR")
        task = await c.get_task(result.task_id)
        listed = await c.list_tasks()
    assert task.id == result.task_id
    assert result.task_id in {t.id for t in listed}


async def test_cancel_and_push_config_management():
    settings = Settings(_env_file=None, allow_private_push_urls=True)  # type: ignore[call-arg]
    app = create_agent_app(BATCH_SPEC, BatchExecutor(), settings, public_url=BASE_URL)
    slow = [new_data_part({"job": "hash", "items": list("abcdefgh"), "step_delay_seconds": 0.3})]
    async with serve(app) as http, _connect(http, streaming=False) as c:
        started = await c.send(slow, return_immediately=True)
        assert started.task_id
        await asyncio.sleep(0.1)

        created = await c.set_push_config(
            started.task_id, "https://hooks.example.com/a2a", token="t", config_id="cfg-1"
        )
        assert created.id == "cfg-1"
        assert [x.id for x in await c.list_push_configs(started.task_id)] == ["cfg-1"]
        await c.delete_push_config(started.task_id, "cfg-1")
        assert await c.list_push_configs(started.task_id) == []

        cancelled = await c.cancel(started.task_id)
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED


async def test_api_key_is_sent_with_every_request():
    settings = Settings(_env_file=None, auth_mode="api_key", api_keys="svc:k1")  # type: ignore[call-arg]
    app = create_agent_app(BATCH_SPEC, BatchExecutor(), settings, public_url=BASE_URL)
    async with serve(app) as http:
        client = await AgentClient.connect(BASE_URL, http=http, api_key="k1", streaming=False)
        result = await client.send("hash\nabc")
        assert result.succeeded
        with pytest.raises(Exception, match="401|nauthorized"):
            bad = await AgentClient.connect(BASE_URL, http=http, api_key="wrong", streaming=False)
            await bad.send("hash\nabc")


def _connect(http, **kwargs):
    class _Ctx:
        async def __aenter__(self):
            self.client = await AgentClient.connect(BASE_URL, http=http, **kwargs)
            return self.client

        async def __aexit__(self, *exc):
            await self.client.close()

    return _Ctx()
