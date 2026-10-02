"""Lazy catalog of hostable agents (framework-specific imports stay optional)."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from a2a.server.agent_execution import AgentExecutor

from agentmesh.config import Settings
from agentmesh.runtime.spec import AgentSpec

AgentBundle = tuple[AgentSpec, AgentExecutor]

# slug -> "module:function"; each function takes Settings and returns (spec, executor).
_REGISTRY: dict[str, str] = {
    "fx": "agentmesh.agents.fx:build_agent",
    "orchestrator": "agentmesh.orchestrator:build_agent",
    "documents": "agentmesh.agents.documents:build_agent",
    "batch": "agentmesh.agents.batch:build_agent",
    "analytics": "agentmesh.agents.analytics:build_agent",  # Google ADK  (extra: adk)
    "quant": "agentmesh.agents.quant:build_agent",  # Microsoft Agent Framework  (extra: maf)
}


class UnknownAgentError(KeyError):
    pass


def available_agents() -> list[str]:
    return sorted(_REGISTRY)


def load_agent(slug: str, settings: Settings) -> AgentBundle:
    try:
        target = _REGISTRY[slug]
    except KeyError:
        raise UnknownAgentError(
            f"unknown agent {slug!r}; available: {', '.join(available_agents())}"
        ) from None
    module_name, _, func_name = target.partition(":")
    factory: Callable[[Settings], AgentBundle] = getattr(
        importlib.import_module(module_name), func_name
    )
    return factory(settings)
