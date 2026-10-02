"""Batch processing agent."""

from __future__ import annotations

from agentmesh.agents.batch.executor import BATCH_SPEC, BatchExecutor
from agentmesh.config import Settings
from agentmesh.runtime.spec import AgentSpec

__all__ = ["BATCH_SPEC", "BatchExecutor", "build_agent"]


def build_agent(_settings: Settings) -> tuple[AgentSpec, BatchExecutor]:
    return BATCH_SPEC, BatchExecutor()
