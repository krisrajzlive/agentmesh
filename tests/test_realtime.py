"""Streaming behaviour that needs real sockets: SSE progress, resubscribe, cancel mid-stream."""

from __future__ import annotations

import asyncio

import pytest
from a2a.helpers import new_data_part

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.client import AgentClient, TaskResult, apply_event
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from tests.live import serve_live

SLOW_JOB = [new_data_part({"job": "hash", "items": list("abcde"), "step_delay_seconds": 0.25})]


@pytest.fixture
def batch_url():
    config = Settings(_env_file=None)  # type: ignore[call-arg]
    with serve_live(
        lambda url: create_agent_app(BATCH_SPEC, BatchExecutor(), config, public_url=url)
    ) as url:
        yield url


async def test_progress_is_streamed_incrementally(batch_url):
    async with await AgentClient.connect(batch_url, streaming=True) as client:
        result = TaskResult()
        seen_states: list[str] = []
        arrival_times: list[float] = []
        loop = asyncio.get_running_loop()
        async for event in client.stream(SLOW_JOB):
            apply_event(result, event)
            seen_states.append(result.state_name)
            arrival_times.append(loop.time())
    assert result.state_name == "completed"
    assert "working" in seen_states
    # With a 0.25s step delay, a truly streamed response spans most of the 1.25s job rather
    # than arriving in one burst at the end.
    assert arrival_times[-1] - arrival_times[0] > 0.8


async def test_a_second_client_can_resubscribe_to_a_running_task(batch_url):
    # A non-streaming client honours return_immediately; a streaming one follows to the end.
    async with await AgentClient.connect(batch_url, streaming=False) as starter:
        started = await starter.send(SLOW_JOB, return_immediately=True)
        assert started.task_id and not started.is_terminal

        async with await AgentClient.connect(batch_url, streaming=True) as watcher:
            result = TaskResult()
            async for event in watcher.subscribe(started.task_id):
                apply_event(result, event)
    assert result.state_name == "completed"
    assert result.artifact("hash-results") is not None
    assert len(result.artifact("hash-results").data) == 5


async def test_cancelling_a_task_stops_it_promptly(batch_url):
    async with await AgentClient.connect(batch_url, streaming=False) as client:
        started = await client.send(
            [
                new_data_part(
                    {"job": "hash", "items": list("abcdefghij"), "step_delay_seconds": 1.0}
                )
            ],
            return_immediately=True,
        )
        await asyncio.sleep(0.2)
        task = await client.cancel(started.task_id)
        await asyncio.sleep(0.3)
        after = await client.get_task(started.task_id)
    assert task.status.state == after.status.state
    assert TaskResult().state != after.status.state  # a concrete state
    assert after.status.state == 5  # TASK_STATE_CANCELED
