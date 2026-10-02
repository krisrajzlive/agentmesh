"""Batch agent: long-running jobs with progress streaming, chunked artifacts and cancellation."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from typing import Any

from a2a.helpers import get_data_parts, get_text_parts, new_data_part
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill

from agentmesh.runtime import AgentSpec, MeshAgentExecutor, TaskInputError

MAX_ITEMS = 500
MAX_STEP_DELAY = 5.0

BATCH_SPEC = AgentSpec(
    slug="batch",
    name="Batch Processing Agent",
    description=(
        "Runs long-running batch jobs over a list of items, streaming progress and partial "
        "results. Supports cancellation and push-notification delivery of the final state."
    ),
    input_modes=["application/json", "text/plain"],
    output_modes=["application/json"],
    skills=[
        AgentSkill(
            id="batch-hash",
            name="Batch hashing",
            description="SHA-256 digest of each item.",
            tags=["batch", "integrity"],
            examples=['{"job": "hash", "items": ["a", "b"]}'],
        ),
        AgentSkill(
            id="batch-wordcount",
            name="Batch word count",
            description="Word and character counts for each item.",
            tags=["batch", "text"],
            examples=['{"job": "wordcount", "items": ["hello world"]}'],
        ),
    ],
)

JOBS: dict[str, Callable[[str], dict[str, Any]]] = {
    "hash": lambda item: {"sha256": hashlib.sha256(item.encode()).hexdigest()},
    "wordcount": lambda item: {"words": len(item.split()), "characters": len(item)},
}


def _parse_request(context: RequestContext) -> tuple[str, list[str], float]:
    message = context.message
    assert message is not None
    for data in get_data_parts(message.parts):
        if isinstance(data, dict) and "job" in data:
            job = str(data["job"])
            items = data.get("items") or []
            delay = float(data.get("step_delay_seconds", 0.0))
            break
    else:
        # Text form: first line is the job name, remaining lines are the items.
        lines = "\n".join(get_text_parts(message.parts)).strip().splitlines()
        if not lines:
            raise TaskInputError("Provide a job, e.g. {'job': 'hash', 'items': ['a', 'b']}.")
        job, items, delay = lines[0].strip().rstrip(":").lower(), lines[1:], 0.0

    if job not in JOBS:
        raise TaskInputError(f"Unknown job {job!r}; available: {', '.join(sorted(JOBS))}.")
    if not isinstance(items, list) or not items:
        raise TaskInputError("The job needs a non-empty list of items.")
    if len(items) > MAX_ITEMS:
        raise TaskInputError(f"At most {MAX_ITEMS} items per job.")
    return job, [str(i) for i in items], min(max(delay, 0.0), MAX_STEP_DELAY)


class BatchExecutor(MeshAgentExecutor):
    agent_name = "batch"

    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        job, items, delay = _parse_request(context)
        handler = JOBS[job]
        total = len(items)
        artifact_id = f"{updater.task_id}-results"

        await updater.start_work(self.text(updater, f"Started {job} job: {total} item(s)"))
        for index, item in enumerate(items, start=1):
            if delay:
                await asyncio.sleep(delay)  # cancellation point; models slow per-item work
            await updater.add_artifact(
                [new_data_part({"index": index, "item": item, **handler(item)})],
                artifact_id=artifact_id,
                name=f"{job}-results",
                append=index > 1,
                last_chunk=index == total,
            )
            await updater.start_work(self.text(updater, f"Processed {index}/{total}"))
        await updater.complete(self.text(updater, f"Completed {job} job: {total} item(s)"))
