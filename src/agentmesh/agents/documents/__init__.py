"""Document processing agent."""

from __future__ import annotations

from agentmesh.agents.documents.executor import DOCUMENTS_SPEC, DocumentsExecutor
from agentmesh.config import Settings
from agentmesh.llm import build_llm
from agentmesh.runtime.spec import AgentSpec

__all__ = ["DOCUMENTS_SPEC", "DocumentsExecutor", "build_agent"]


def build_agent(settings: Settings) -> tuple[AgentSpec, DocumentsExecutor]:
    return DOCUMENTS_SPEC, DocumentsExecutor(build_llm(settings))
