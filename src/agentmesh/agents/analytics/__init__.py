"""Google ADK text-analytics agent (requires the ``adk`` extra)."""

from agentmesh.agents.analytics.agent import (
    ANALYTICS_SPEC,
    build_agent,
    build_analytics_agent,
    build_analytics_executor,
)

__all__ = ["ANALYTICS_SPEC", "build_agent", "build_analytics_agent", "build_analytics_executor"]
