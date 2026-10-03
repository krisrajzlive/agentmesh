from __future__ import annotations

from decimal import Decimal

import pytest

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.client import AgentClient
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, serve


def make_fx(settings: Settings):
    executor = build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")}))
    return create_agent_app(FX_SPEC, executor, settings, public_url=BASE_URL)


@pytest.fixture
def db_settings(tmp_path) -> Settings:
    url = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"
    return Settings(_env_file=None, task_store_url=url)  # type: ignore[call-arg]


async def test_tasks_survive_an_agent_restart(db_settings):
    async with serve(make_fx(db_settings)) as http:
        client = await AgentClient.connect(BASE_URL, http=http, streaming=False)
        done = await client.send("Convert 10 USD to EUR")
        assert done.succeeded

    # A brand-new process (new app, new engine) sees the task written by the old one.
    async with serve(make_fx(db_settings)) as http:
        client = await AgentClient.connect(BASE_URL, http=http, streaming=False)
        task = await client.get_task(done.task_id)
        listed = await client.list_tasks()
    assert task.id == done.task_id
    assert done.task_id in {t.id for t in listed}


async def test_agents_sharing_a_database_do_not_see_each_others_tasks(db_settings):
    fx = make_fx(db_settings)
    batch = create_agent_app(BATCH_SPEC, BatchExecutor(), db_settings, public_url=BASE_URL)
    async with serve(fx) as fx_http, serve(batch) as batch_http:
        fx_client = await AgentClient.connect(BASE_URL, http=fx_http, streaming=False)
        batch_client = await AgentClient.connect(BASE_URL, http=batch_http, streaming=False)
        await fx_client.send("Convert 10 USD to EUR")
        await batch_client.send("hash\nabc")

        assert len(await fx_client.list_tasks()) == 1
        assert len(await batch_client.list_tasks()) == 1
