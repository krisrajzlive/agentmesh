from __future__ import annotations

from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageConfiguration, SendMessageRequest, StreamResponse


def user_request(
    text: str,
    *,
    task_id: str | None = None,
    context_id: str | None = None,
    configuration: SendMessageConfiguration | None = None,
) -> SendMessageRequest:
    message = new_text_message(text, role=Role.ROLE_USER, task_id=task_id, context_id=context_id)
    request = SendMessageRequest(message=message)
    if configuration is not None:
        request.configuration.CopyFrom(configuration)
    return request


def final_state(events: list[StreamResponse]) -> str:
    """State name of the last status carried by a stream of responses."""
    from a2a.types import TaskState

    state = None
    for event in events:
        if event.HasField("task"):
            state = event.task.status.state
        elif event.HasField("status_update"):
            state = event.status_update.status.state
    return TaskState.Name(state) if state is not None else "NONE"


def task_id_of(events: list[StreamResponse]) -> str:
    for event in events:
        if event.HasField("task"):
            return event.task.id
        if event.HasField("status_update"):
            return event.status_update.task_id
    raise AssertionError("no task in stream")


def context_id_of(events: list[StreamResponse]) -> str:
    for event in events:
        if event.HasField("task"):
            return event.task.context_id
        if event.HasField("status_update"):
            return event.status_update.context_id
    raise AssertionError("no context in stream")
