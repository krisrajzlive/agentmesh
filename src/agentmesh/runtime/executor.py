"""Base executor for native AgentMesh agents."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import Callable

from a2a.helpers import new_task, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState

from agentmesh.observability.logging import get_logger
from agentmesh.observability.metrics import TASKS_TOTAL

log = get_logger(__name__)


class TaskInputError(Exception):
    """The request cannot be served; the task is *rejected* with this message."""


class MeshAgentExecutor(AgentExecutor):
    """Template for native agents.

    Subclasses implement :meth:`run`. This base class owns the A2A lifecycle plumbing:
    task creation, ``working`` / ``failed`` / ``canceled`` transitions, error containment,
    cooperative cancellation and metrics.
    """

    agent_name = "agent"

    def __init__(self) -> None:
        self._running: dict[str, asyncio.Task[None]] = {}

    @abstractmethod
    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Do the work. Must finish by calling ``updater.complete()`` / ``requires_input()``."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            raise TaskInputError("request carries no message")
        task = context.current_task
        if task is None:
            task = new_task(
                context.task_id or message.task_id,
                context.context_id or message.context_id,
                TaskState.TASK_STATE_SUBMITTED,
                history=[message],
            )
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        current = asyncio.current_task()
        if current is not None:
            self._running[task.id] = current
        try:
            await self.run(context, updater)
            TASKS_TOTAL.labels(self.agent_name, "ok").inc()
        except TaskInputError as exc:
            TASKS_TOTAL.labels(self.agent_name, "rejected").inc()
            await updater.reject(self.text(updater, str(exc)))
        except asyncio.CancelledError:
            TASKS_TOTAL.labels(self.agent_name, "canceled").inc()
            raise
        except Exception:
            log.exception("agent_task_failed", agent=self.agent_name, task_id=task.id)
            TASKS_TOTAL.labels(self.agent_name, "error").inc()
            await updater.failed(self.text(updater, "The agent hit an internal error."))
        finally:
            self._running.pop(task.id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        if task_id is None:
            return
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
        updater = TaskUpdater(event_queue, task_id, context.context_id or "")
        await updater.cancel()

    @staticmethod
    def text(updater: TaskUpdater, content: str) -> Message:
        """Agent message carrying ``content``, correlated to the current task."""
        return new_text_message(content, context_id=updater.context_id, task_id=updater.task_id)


ExecutorFactory = Callable[[], AgentExecutor]
