"""Multi-agent orchestration."""

from __future__ import annotations

import httpx

from agentmesh.config import Settings
from agentmesh.llm import LLMProvider, build_llm
from agentmesh.orchestrator.directory import AgentDirectory, RegistryDirectory, StaticDirectory
from agentmesh.orchestrator.executor import ORCHESTRATOR_SPEC, OrchestratorExecutor
from agentmesh.orchestrator.planner import LLMPlanner, Planner, PlanningError, RuleBasedPlanner
from agentmesh.registry.client import RegistryClient
from agentmesh.runtime.spec import AgentSpec

__all__ = [
    "ORCHESTRATOR_SPEC",
    "AgentDirectory",
    "LLMPlanner",
    "OrchestratorExecutor",
    "Planner",
    "PlanningError",
    "RegistryDirectory",
    "RuleBasedPlanner",
    "StaticDirectory",
    "build_agent",
    "build_orchestrator",
]


def build_orchestrator(
    settings: Settings,
    *,
    directory: AgentDirectory | None = None,
    llm: LLMProvider | None = None,
    http: httpx.AsyncClient | None = None,
) -> OrchestratorExecutor:
    if directory is None:
        if settings.registry_url:
            key = (
                settings.outbound_api_key.get_secret_value() if settings.outbound_api_key else None
            )
            token = (
                settings.outbound_bearer_token.get_secret_value()
                if settings.outbound_bearer_token
                else None
            )
            directory = RegistryDirectory(
                RegistryClient(settings.registry_url, api_key=key, bearer_token=token)
            )
        elif settings.static_agent_urls:
            directory = StaticDirectory(settings.static_agent_urls)
        else:
            raise RuntimeError(
                "orchestrator needs AGENTMESH_REGISTRY_URL or AGENTMESH_STATIC_AGENT_URLS"
            )
    llm = llm or build_llm(settings)
    planner: Planner = LLMPlanner(llm) if llm else RuleBasedPlanner()
    return OrchestratorExecutor(directory, planner, settings, http=http)


def build_agent(settings: Settings) -> tuple[AgentSpec, OrchestratorExecutor]:
    return ORCHESTRATOR_SPEC, build_orchestrator(settings)
