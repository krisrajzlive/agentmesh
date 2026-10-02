"""Quantitative Analysis agent built on Microsoft Agent Framework, served via AgentMesh."""

from __future__ import annotations

from typing import Any

from a2a.types import AgentSkill
from agent_framework import Agent, SupportsChatGetResponse
from agent_framework.a2a import A2AExecutor
from agent_framework.openai import OpenAIChatCompletionClient

from agentmesh.agents.endpoint import resolve_endpoint
from agentmesh.agents.quant.tools import compound_growth, descriptive_statistics, loan_payment
from agentmesh.config import Settings
from agentmesh.runtime.spec import AgentSpec

QUANT_SPEC = AgentSpec(
    slug="quant",
    name="Quantitative Analysis Agent",
    description=(
        "Descriptive statistics, loan amortisation and compound-growth calculations. "
        "Implemented with Microsoft Agent Framework and exposed over A2A."
    ),
    framework="microsoft-agent-framework",
    skills=[
        AgentSkill(
            id="descriptive-statistics",
            name="Descriptive statistics",
            description="Mean, median, spread and range of a list of numbers.",
            tags=["math", "statistics"],
            examples=["Summarise 12, 15, 9, 22, 18"],
        ),
        AgentSkill(
            id="loan-payment",
            name="Loan payment",
            description="Monthly payment and total interest for a fixed-rate loan.",
            tags=["finance", "loans"],
            examples=["Monthly payment on 250000 at 6.5% over 30 years?"],
        ),
        AgentSkill(
            id="compound-growth",
            name="Compound growth",
            description="Future value of an investment with periodic compounding.",
            tags=["finance", "investing"],
            examples=["What does 10000 grow to at 7% over 20 years?"],
        ),
    ],
)

INSTRUCTIONS = (
    "You are a quantitative analyst. Use the provided tools for every calculation; never do "
    "arithmetic yourself. State inputs, results and units clearly. If an input is missing, ask."
)


def build_quant_agent(client: SupportsChatGetResponse[Any]) -> Agent[Any]:
    return Agent(
        client=client,
        name="quant",
        description=QUANT_SPEC.description,
        instructions=INSTRUCTIONS,
        tools=[descriptive_statistics, loan_payment, compound_growth],
    )


def build_quant_executor(client: SupportsChatGetResponse[Any]) -> A2AExecutor:
    return A2AExecutor(build_quant_agent(client), stream=True)


def build_agent(settings: Settings) -> tuple[AgentSpec, A2AExecutor]:
    endpoint = resolve_endpoint(settings)
    client = OpenAIChatCompletionClient(
        model=endpoint.model, api_key=endpoint.api_key, base_url=endpoint.base_url
    )
    return QUANT_SPEC, build_quant_executor(client)
