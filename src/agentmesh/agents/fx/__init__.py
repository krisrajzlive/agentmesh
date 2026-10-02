"""FX conversion agent."""

from __future__ import annotations

from agentmesh.agents.fx.executor import FX_SPEC, FxExecutor
from agentmesh.agents.fx.parser import LLMParser, RuleBasedParser
from agentmesh.agents.fx.rates import FrankfurterRates, RateProvider, StaticRates
from agentmesh.config import Settings
from agentmesh.llm import LLMProvider, build_llm
from agentmesh.runtime.spec import AgentSpec

__all__ = [
    "FX_SPEC",
    "FrankfurterRates",
    "FxExecutor",
    "LLMParser",
    "RateProvider",
    "RuleBasedParser",
    "StaticRates",
    "build_agent",
    "build_fx_executor",
]


def build_fx_executor(
    settings: Settings, *, rates: RateProvider | None = None, llm: LLMProvider | None = None
) -> FxExecutor:
    llm = llm or build_llm(settings)
    parser = LLMParser(llm) if llm else RuleBasedParser()
    return FxExecutor(parser, rates or FrankfurterRates())


def build_agent(settings: Settings) -> tuple[AgentSpec, FxExecutor]:
    return FX_SPEC, build_fx_executor(settings)
