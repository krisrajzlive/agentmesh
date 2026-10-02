"""Orchestrator: plans a request over registered agents and runs the plan to completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from a2a.helpers import get_message_text, new_data_part, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill, Part, TaskState
from google.protobuf.json_format import MessageToDict

from agentmesh.client import AgentClient, TaskResult
from agentmesh.config import Settings
from agentmesh.observability.logging import get_logger
from agentmesh.orchestrator.directory import AgentDirectory
from agentmesh.orchestrator.planner import Planner, PlanningError, PlanStep
from agentmesh.registry.models import AgentRecord
from agentmesh.resilience import CircuitBreaker
from agentmesh.runtime import AgentSpec, MeshAgentExecutor, TaskInputError

log = get_logger(__name__)

STATE_KEY = "agentmesh.orchestrator"

ORCHESTRATOR_SPEC = AgentSpec(
    slug="orchestrator",
    name="Orchestrator Agent",
    description=(
        "Routes a request to the best specialist agents in the mesh, runs independent steps in "
        "parallel, chains dependent ones and aggregates the results. Works across agents built "
        "with any framework."
    ),
    # Files are forwarded to the selected agent, so accept every type the mesh handles.
    input_modes=["text/plain", "text/markdown", "text/csv", "application/json"],
    skills=[
        AgentSkill(
            id="multi-agent-orchestration",
            name="Multi-agent orchestration",
            description="Decompose a request and delegate to specialist agents.",
            tags=["orchestration", "routing", "multi-agent"],
            examples=[
                "Convert 100 USD to EUR; summarise the numbers 4, 8, 15, 16, 23, 42",
                "Redact personal data from this text, then count its words",
            ],
        )
    ],
)


def _is_transient(exc: BaseException | None) -> bool:
    """True for network failures, including ones the SDK wraps in its own error types."""
    while exc is not None:
        if isinstance(exc, httpx.TransportError | TimeoutError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@dataclass(slots=True)
class StepOutcome:
    step_id: str
    agent_id: str
    ok: bool
    text: str
    state: str = ""
    result: TaskResult | None = None
    needs_input: bool = False


@dataclass(slots=True)
class RunState:
    """Everything needed to resume an orchestration after an input-required pause."""

    steps: list[PlanStep]
    done: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "id": s.id,
                    "agent": s.agent_id,
                    "instruction": s.instruction,
                    "depends_on": list(s.depends_on),
                }
                for s in self.steps
            ],
            "done": self.done,
            "pending": self.pending or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        steps = [
            PlanStep(s["id"], s["agent"], s["instruction"], tuple(s.get("depends_on", [])))
            for s in data["steps"]
        ]
        return cls(
            steps=steps, done=dict(data.get("done", {})), pending=data.get("pending") or None
        )


class OrchestratorExecutor(MeshAgentExecutor):
    agent_name = "orchestrator"

    def __init__(
        self,
        directory: AgentDirectory,
        planner: Planner,
        settings: Settings,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self._directory = directory
        self._planner = planner
        self._settings = settings
        self._http = http  # injected in tests; production clients own their connections
        self._breakers: dict[str, CircuitBreaker] = {}

    async def aclose(self) -> None:
        await self._directory.aclose()

    # ------------------------------------------------------------------ main flow

    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        message = context.message
        assert message is not None
        request_text = get_message_text(message).strip()
        agents = {a.id: a for a in await self._directory.agents()}

        state = self._saved_state(context)
        if state is None:
            if not request_text:
                raise TaskInputError("Describe what you need done.")
            try:
                steps = await self._planner.plan(request_text, list(agents.values()))
            except PlanningError as exc:
                raise TaskInputError(str(exc)) from exc
            state = RunState(steps=steps)
            await updater.start_work(
                self.text(updater, "Plan: " + " | ".join(f"{s.id}->{s.agent_id}" for s in steps))
            )
            extra = [p for p in message.parts if not p.HasField("text")]
        else:
            extra = []
            if not request_text:
                raise TaskInputError("Please answer the agent's question to continue.")
            await self._resume(state, request_text, agents, updater)
            if state.pending:
                await self._pause(state, updater)
                return

        await self._execute(state, agents, updater, extra)

    async def _execute(
        self,
        state: RunState,
        agents: dict[str, AgentRecord],
        updater: TaskUpdater,
        extra_parts: list[Part],
    ) -> None:
        first_step_id = state.steps[0].id if state.steps else ""
        while True:
            ready = [
                s
                for s in state.steps
                if s.id not in state.done and all(d in state.done for d in s.depends_on)
            ]
            if not ready:
                break
            outcomes = await asyncio.gather(
                *(
                    self._run_step(
                        s,
                        agents,
                        state,
                        extra_parts if (s.id == first_step_id or len(state.steps) == 1) else [],
                    )
                    for s in ready
                )
            )
            for outcome in outcomes:
                await self._record(state, outcome, updater)
            if state.pending:
                await self._pause(state, updater)
                return
        await self._finish(state, updater)

    # ------------------------------------------------------------------ steps

    async def _run_step(
        self,
        step: PlanStep,
        agents: dict[str, AgentRecord],
        state: RunState,
        extra_parts: list[Part],
    ) -> StepOutcome:
        agent = agents.get(step.agent_id)
        if agent is None:
            return StepOutcome(step.id, step.agent_id, False, "agent is no longer available")
        instruction = step.instruction
        if step.depends_on:
            context_lines = [
                f"- {dep}: {state.done[dep].get('text', '')}" for dep in step.depends_on
            ]
            instruction += "\n\nResults from earlier steps:\n" + "\n".join(context_lines)
        parts = [new_text_part(instruction), *extra_parts]
        return await self._call(step.id, agent, parts, task_id=None, context_id=None)

    async def _call(
        self,
        step_id: str,
        agent: AgentRecord,
        parts: list[Part],
        *,
        task_id: str | None,
        context_id: str | None,
    ) -> StepOutcome:
        breaker = self._breakers.setdefault(
            agent.url, CircuitBreaker(f"agent:{agent.id}", failure_threshold=3, reset_timeout=30)
        )
        if not breaker.allow():
            return StepOutcome(step_id, agent.id, False, "agent circuit is open; try again shortly")

        last_error = ""
        for attempt in range(2):
            try:
                async with asyncio.timeout(self._settings.agent_call_timeout_seconds):
                    client = await AgentClient.connect(
                        agent.url,
                        api_key=self._key(),
                        bearer_token=self._bearer(),
                        headers=self._trace_headers(),
                        http=self._http,
                    )
                    try:
                        result = await client.send(parts, task_id=task_id, context_id=context_id)
                    finally:
                        await client.close()
            except Exception as exc:
                if _is_transient(exc):
                    last_error = str(exc) or type(exc).__name__
                    log.warning(
                        "agent_call_retry", agent=agent.id, attempt=attempt + 1, error=last_error
                    )
                    continue
                breaker.record_failure()
                log.warning("agent_call_failed", agent=agent.id, error=str(exc))
                return StepOutcome(step_id, agent.id, False, f"call failed: {exc}")
            breaker.record_success()
            return self._outcome(step_id, agent, result)
        breaker.record_failure()
        return StepOutcome(step_id, agent.id, False, f"agent unreachable ({last_error})")

    @staticmethod
    def _outcome(step_id: str, agent: AgentRecord, result: TaskResult) -> StepOutcome:
        if result.needs_input:
            return StepOutcome(
                step_id,
                agent.id,
                False,
                result.output_text,
                result.state_name,
                result,
                needs_input=True,
            )
        ok = result.succeeded
        text = result.output_text or ("" if ok else f"agent finished in state {result.state_name}")
        return StepOutcome(step_id, agent.id, ok, text, result.state_name, result)

    # ------------------------------------------------------- recording and pausing

    async def _record(self, state: RunState, outcome: StepOutcome, updater: TaskUpdater) -> None:
        if outcome.needs_input and outcome.result is not None:
            state.pending = {
                "step_id": outcome.step_id,
                "agent_id": outcome.agent_id,
                "task_id": outcome.result.task_id,
                "context_id": outcome.result.context_id,
                "question": outcome.text,
            }
            return
        state.done[outcome.step_id] = {
            "agent": outcome.agent_id,
            "ok": outcome.ok,
            "text": outcome.text,
            "state": outcome.state,
        }
        parts = [new_text_part(outcome.text)] if outcome.text else []
        if outcome.result is not None:
            for artifact in outcome.result.artifacts:
                if artifact.text and artifact.text != outcome.text:
                    parts.append(new_text_part(artifact.text))
                parts.extend(new_data_part(d) for d in artifact.data)
        if parts:
            await updater.add_artifact(parts, name=f"{outcome.step_id}:{outcome.agent_id}")
        marker = "done" if outcome.ok else "failed"
        await updater.start_work(
            self.text(updater, f"Step {outcome.step_id} ({outcome.agent_id}) {marker}")
        )

    async def _pause(self, state: RunState, updater: TaskUpdater) -> None:
        assert state.pending is not None
        question = f"{state.pending['agent_id']} asks: {state.pending['question']}"
        message = updater.new_agent_message(
            [new_text_part(question)], metadata={STATE_KEY: state.to_dict()}
        )
        await updater.requires_input(message)

    def _saved_state(self, context: RequestContext) -> RunState | None:
        task = context.current_task
        if task is None or task.status.state != TaskState.TASK_STATE_INPUT_REQUIRED:
            return None
        if not task.status.HasField("message"):
            return None
        metadata = MessageToDict(task.status.message.metadata)
        saved = metadata.get(STATE_KEY)
        return RunState.from_dict(saved) if saved else None

    async def _resume(
        self,
        state: RunState,
        reply: str,
        agents: dict[str, AgentRecord],
        updater: TaskUpdater,
    ) -> None:
        pending = state.pending
        state.pending = None
        if not pending:
            return
        agent = agents.get(pending["agent_id"])
        if agent is None:
            outcome = StepOutcome(pending["step_id"], pending["agent_id"], False, "agent is gone")
        else:
            outcome = await self._call(
                pending["step_id"],
                agent,
                [new_text_part(reply)],
                task_id=pending["task_id"] or None,
                context_id=pending["context_id"] or None,
            )
        await self._record(state, outcome, updater)

    # ------------------------------------------------------------------ finishing

    async def _finish(self, state: RunState, updater: TaskUpdater) -> None:
        total = len(state.steps)
        ok = [s for s in state.steps if state.done.get(s.id, {}).get("ok")]
        lines = []
        for step in state.steps:
            outcome = state.done.get(step.id, {})
            label = "" if outcome.get("ok") else " (failed)"
            lines.append(f"- {step.agent_id}{label}: {outcome.get('text') or 'no output'}")
        summary = f"Completed {len(ok)} of {total} step(s).\n" + "\n".join(lines)
        await updater.add_artifact(
            [
                new_text_part(summary),
                new_data_part(
                    {
                        "steps": [
                            {
                                "id": s.id,
                                "agent": s.agent_id,
                                "ok": bool(state.done.get(s.id, {}).get("ok")),
                                "state": state.done.get(s.id, {}).get("state", ""),
                            }
                            for s in state.steps
                        ]
                    }
                ),
            ],
            name="summary",
        )
        message = self.text(updater, summary)
        if ok:
            await updater.complete(message)
        else:
            await updater.failed(message)

    # ------------------------------------------------------------------- helpers

    def _key(self) -> str | None:
        key = self._settings.outbound_api_key
        return key.get_secret_value() if key else None

    def _bearer(self) -> str | None:
        token = self._settings.outbound_bearer_token
        return token.get_secret_value() if token else None

    @staticmethod
    def _trace_headers() -> dict[str, str]:
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        return {"x-request-id": str(request_id)} if request_id else {}
