"""Fold A2A stream events into a simple, typed result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from a2a.helpers import get_message_text
from a2a.types import Artifact, Message, StreamResponse, Task, TaskState
from google.protobuf.json_format import MessageToDict

INTERRUPTED = {TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED}
TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


@dataclass(slots=True)
class ArtifactData:
    artifact_id: str
    name: str
    text: str
    data: list[Any]

    @classmethod
    def from_proto(cls, artifact: Artifact) -> ArtifactData:
        texts = [p.text for p in artifact.parts if p.HasField("text")]
        data = [MessageToDict(p.data) for p in artifact.parts if p.HasField("data")]
        return cls(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            text="\n".join(texts),
            data=data,
        )


@dataclass(slots=True)
class TaskResult:
    """Snapshot of a task as seen by the caller."""

    task_id: str = ""
    context_id: str = ""
    state: TaskState = TaskState.TASK_STATE_UNSPECIFIED
    text: str = ""
    artifacts: list[ArtifactData] = field(default_factory=list)
    reply: Message | None = None  # set when the agent answered with a bare Message

    @property
    def state_name(self) -> str:
        name: str = TaskState.Name(self.state)
        return name.removeprefix("TASK_STATE_").lower()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def needs_input(self) -> bool:
        return self.state in INTERRUPTED

    @property
    def succeeded(self) -> bool:
        return self.state == TaskState.TASK_STATE_COMPLETED or (
            self.reply is not None and self.state == TaskState.TASK_STATE_UNSPECIFIED
        )

    def artifact(self, name: str) -> ArtifactData | None:
        return next((a for a in self.artifacts if a.name == name), None)


def apply_event(result: TaskResult, event: StreamResponse) -> TaskResult:
    """Update ``result`` in place with one streamed event and return it."""
    if event.HasField("task"):
        _apply_task(result, event.task)
    elif event.HasField("message"):
        result.reply = event.message
        result.text = get_message_text(event.message)
        result.task_id = event.message.task_id or result.task_id
        result.context_id = event.message.context_id or result.context_id
    elif event.HasField("status_update"):
        update = event.status_update
        result.task_id, result.context_id = update.task_id, update.context_id
        result.state = update.status.state
        if update.status.HasField("message"):
            result.text = get_message_text(update.status.message)
    elif event.HasField("artifact_update"):
        _apply_artifact(result, event.artifact_update.artifact, event.artifact_update.append)
    return result


def _apply_task(result: TaskResult, task: Task) -> None:
    result.task_id, result.context_id = task.id, task.context_id
    result.state = task.status.state
    if task.status.HasField("message"):
        result.text = get_message_text(task.status.message)
    if task.artifacts:
        result.artifacts = [ArtifactData.from_proto(a) for a in task.artifacts]


def _apply_artifact(result: TaskResult, artifact: Artifact, append: bool) -> None:
    incoming = ArtifactData.from_proto(artifact)
    existing = next((a for a in result.artifacts if a.artifact_id == incoming.artifact_id), None)
    if existing is None:
        result.artifacts.append(incoming)
    elif append:
        existing.text = "\n".join(t for t in (existing.text, incoming.text) if t)
        existing.data.extend(incoming.data)
    else:
        existing.name, existing.text, existing.data = incoming.name, incoming.text, incoming.data
