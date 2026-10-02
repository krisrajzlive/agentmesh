"""Turn a user request into an executable plan over the available agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from agentmesh.llm import ChatMessage, LLMError, LLMProvider
from agentmesh.observability.logging import get_logger
from agentmesh.registry.models import AgentRecord

log = get_logger(__name__)

MAX_STEPS = 8


class PlanningError(Exception):
    """No acceptable plan could be produced."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    id: str
    agent_id: str
    instruction: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)


class Planner(Protocol):
    async def plan(self, request: str, agents: list[AgentRecord]) -> list[PlanStep]: ...


_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset(
    "the and for with that this from into over your you can are was were will would could "
    "please give get show tell what how much many convert calculate compute".split()
)
# "a; b" -> independent steps. "a then b" / "a and then b" -> b depends on a.
_SPLIT = re.compile(r"\s*(;|\n|\band then\b|\bthen\b)\s*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _score(clause_tokens: set[str], agent: AgentRecord) -> float:
    stems = {_stem(t) for t in clause_tokens}
    total = 0.0
    for skill in agent.skills:
        strong = {_stem(t) for t in _tokens(f"{skill.name} {' '.join(skill.tags)}")}
        weak = {_stem(t) for t in _tokens(f"{skill.description} {' '.join(skill.examples)}")}
        total = max(total, 2.0 * len(stems & strong) + 1.0 * len(stems & (weak - strong)))
    return total


def split_clauses(request: str) -> list[tuple[str, bool]]:
    """Split into ``(clause, sequential)`` pairs; ``sequential`` follows the previous clause."""
    parts = _SPLIT.split(request.strip())
    clauses: list[tuple[str, bool]] = []
    sequential = False
    for part in parts:
        if not part or not part.strip():
            continue
        lowered = part.strip().lower()
        if lowered in {";", "\n"}:
            sequential = False
        elif lowered in {"then", "and then"}:
            sequential = True
        else:
            clauses.append((part.strip(), sequential and bool(clauses)))
            sequential = False
    return clauses


class RuleBasedPlanner:
    """Deterministic routing by lexical overlap with each agent's skills."""

    def __init__(self, min_score: float = 1.0) -> None:
        self._min_score = min_score

    async def plan(self, request: str, agents: list[AgentRecord]) -> list[PlanStep]:
        if not agents:
            raise PlanningError("no agents are available")
        clauses = split_clauses(request)
        if not clauses:
            raise PlanningError("empty request")
        steps: list[PlanStep] = []
        for index, (clause, sequential) in enumerate(clauses[:MAX_STEPS], start=1):
            tokens = _tokens(clause)
            ranked = sorted(
                ((_score(tokens, a), a) for a in agents), key=lambda p: (-p[0], p[1].id)
            )
            best_score, best = ranked[0]
            if best_score < self._min_score:
                raise PlanningError(f"no available agent can handle: {clause!r}")
            depends = (steps[-1].id,) if sequential and steps else ()
            steps.append(PlanStep(f"s{index}", best.id, clause, depends))
        return steps


_PLANNER_PROMPT = (
    "You route user requests to specialist agents. Reply with JSON only: "
    '{"steps": [{"id": "s1", "agent": "<agent id>", "instruction": "<self-contained task>", '
    '"depends_on": ["<step id>"]}]}. Use only the listed agents. Split the request into the '
    "fewest steps needed. Use depends_on only when a step needs an earlier step's output."
)


class LLMPlanner:
    """LLM-generated plan, validated against the catalog; falls back to rule-based routing."""

    def __init__(self, llm: LLMProvider, fallback: Planner | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or RuleBasedPlanner()

    async def plan(self, request: str, agents: list[AgentRecord]) -> list[PlanStep]:
        catalog = [
            {
                "id": a.id,
                "description": a.description,
                "skills": [{"id": s.id, "description": s.description} for s in a.skills],
            }
            for a in agents
        ]
        try:
            reply = await self._llm.complete(
                [
                    ChatMessage("system", _PLANNER_PROMPT),
                    ChatMessage("user", f"Agents:\n{json.dumps(catalog)}\n\nRequest:\n{request}"),
                ],
                json_mode=True,
                max_tokens=600,
            )
            return self._validate(json.loads(reply.content), agents)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            log.warning("planner_llm_fallback", error=str(exc))
            return await self._fallback.plan(request, agents)

    @staticmethod
    def _validate(payload: dict, agents: list[AgentRecord]) -> list[PlanStep]:  # type: ignore[type-arg]
        known = {a.id for a in agents}
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list) or not 0 < len(raw_steps) <= MAX_STEPS:
            raise ValueError("bad step count")
        steps: list[PlanStep] = []
        seen: set[str] = set()
        for raw in raw_steps:
            step = PlanStep(
                id=str(raw["id"]),
                agent_id=str(raw["agent"]),
                instruction=str(raw["instruction"]).strip(),
                depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
            )
            if step.agent_id not in known or not step.instruction or step.id in seen:
                raise ValueError(f"invalid step {step.id}")
            if any(d not in seen for d in step.depends_on):
                raise ValueError("dependency on unknown or later step")  # also rules out cycles
            seen.add(step.id)
            steps.append(step)
        return steps
