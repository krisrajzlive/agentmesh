"""Microsoft Agent Framework quantitative-analysis agent (requires the ``maf`` extra)."""

from agentmesh.agents.quant.agent import (
    QUANT_SPEC,
    build_agent,
    build_quant_agent,
    build_quant_executor,
)

__all__ = ["QUANT_SPEC", "build_agent", "build_quant_agent", "build_quant_executor"]
