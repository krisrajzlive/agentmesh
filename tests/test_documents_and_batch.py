from __future__ import annotations

import asyncio

import pytest
from a2a.helpers import new_data_part, new_message, new_raw_part, new_text_part
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskState,
)
from google.protobuf.json_format import MessageToDict

from agentmesh.agents.batch import BATCH_SPEC, BatchExecutor
from agentmesh.agents.documents import DOCUMENTS_SPEC, DocumentsExecutor
from agentmesh.agents.documents.processing import extractive_summary, profile_csv
from agentmesh.runtime import create_agent_app
from tests.conftest import BASE_URL, a2a_client, serve
from tests.helpers import final_state, task_id_of

CSV = b"id,price,name\n1,9.5,apple\n2,12,pear\n3,,fig\n"
ARTICLE = (
    "Solar power is growing quickly. Solar panels convert sunlight into electricity. "
    "Costs of solar panels have fallen sharply. Wind power is also growing. "
    "Many countries now invest in solar and wind. Storage remains a challenge."
)


def request(*parts, **config) -> SendMessageRequest:
    req = SendMessageRequest(message=new_message(list(parts), role=Role.ROLE_USER))
    if config:
        req.configuration.CopyFrom(SendMessageConfiguration(**config))
    return req


def artifacts(events) -> list[dict]:
    last = events[-1]
    task = last.task if last.HasField("task") else None
    assert task is not None, "expected the final response to be a task"
    return [MessageToDict(a) for a in task.artifacts]


@pytest.fixture
def documents_app(settings):
    return create_agent_app(DOCUMENTS_SPEC, DocumentsExecutor(), settings, public_url=BASE_URL)


@pytest.fixture
def batch_app(settings):
    return create_agent_app(BATCH_SPEC, BatchExecutor(), settings, public_url=BASE_URL)


def test_extractive_summary_keeps_order_and_length():
    summary = extractive_summary(ARTICLE, 2)
    assert summary.count(".") == 2
    assert "Solar" in summary


def test_csv_profile_infers_types_and_nulls():
    columns = {c["column"]: c for c in profile_csv(CSV.decode())}
    assert columns["id"]["type"] == "integer"
    assert columns["price"]["type"] == "float"
    assert columns["price"]["nulls"] == 1
    assert columns["name"]["type"] == "string"
    assert columns["price"]["max"] == 12.0


async def test_summarises_uploaded_text_file(documents_app):
    part = new_raw_part(ARTICLE.encode(), media_type="text/plain", filename="solar.txt")
    async with serve(documents_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(part))]
    assert final_state(events) == "TASK_STATE_COMPLETED"
    (artifact,) = artifacts(events)
    assert artifact["name"] == "summary"
    assert any("words" in p.get("data", {}) for p in artifact["parts"])


async def test_profiles_uploaded_csv_file(documents_app):
    part = new_raw_part(CSV, media_type="text/csv", filename="items.csv")
    async with serve(documents_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(part))]
    assert final_state(events) == "TASK_STATE_COMPLETED"
    (artifact,) = artifacts(events)
    data = next(p["data"] for p in artifact["parts"] if "data" in p)
    assert [c["column"] for c in data["columns"]] == ["id", "price", "name"]


async def test_structured_data_part_is_accepted(documents_app):
    async with serve(documents_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(new_data_part({"text": ARTICLE})))]
    assert final_state(events) == "TASK_STATE_COMPLETED"


async def test_asks_for_input_when_nothing_to_analyse(documents_app):
    async with serve(documents_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(new_text_part("hello")))]
    assert final_state(events) == "TASK_STATE_INPUT_REQUIRED"


async def test_binary_upload_is_rejected(documents_app):
    part = new_raw_part(b"\xff\xfe\x00\x01", media_type="text/plain")
    async with serve(documents_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(part))]
    assert final_state(events) == "TASK_STATE_REJECTED"


async def test_batch_job_streams_chunked_results(batch_app):
    payload = new_data_part({"job": "hash", "items": ["a", "b", "c"]})
    async with serve(batch_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(payload))]
    assert final_state(events) == "TASK_STATE_COMPLETED"
    (artifact,) = artifacts(events)
    rows = [p["data"] for p in artifact["parts"]]
    assert [r["index"] for r in rows] == [1, 2, 3]
    assert rows[0]["sha256"].startswith("ca978112")  # sha256("a")


async def test_batch_text_form_and_validation(batch_app):
    async with serve(batch_app) as http:
        client = await a2a_client(http, streaming=False)
        ok = [
            e
            async for e in client.send_message(request(new_text_part("wordcount\nhello big world")))
        ]
        bad = [
            e
            async for e in client.send_message(
                request(new_data_part({"job": "nope", "items": ["x"]}))
            )
        ]
    assert final_state(ok) == "TASK_STATE_COMPLETED"
    assert final_state(bad) == "TASK_STATE_REJECTED"


async def test_running_batch_job_can_be_cancelled(batch_app):
    slow = new_data_part({"job": "hash", "items": list("abcdefghij"), "step_delay_seconds": 0.3})
    async with serve(batch_app) as http:
        client = await a2a_client(http, streaming=False)
        events = [e async for e in client.send_message(request(slow, return_immediately=True))]
        task_id = task_id_of(events)
        await asyncio.sleep(0.1)
        await client.cancel_task(CancelTaskRequest(id=task_id))
        task = await client.get_task(GetTaskRequest(id=task_id))
    assert task.status.state == TaskState.TASK_STATE_CANCELED
