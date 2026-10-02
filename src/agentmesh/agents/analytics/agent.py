"""Text Analytics agent built on Google ADK, served through the AgentMesh runtime."""

from __future__ import annotations

from typing import Any

from a2a.types import AgentSkill
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.models import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from agentmesh.agents.analytics.tools import extract_keywords, redact_pii, text_statistics
from agentmesh.agents.endpoint import resolve_endpoint
from agentmesh.config import Settings
from agentmesh.runtime.spec import AgentSpec

APP_NAME = "analytics"

ANALYTICS_SPEC = AgentSpec(
    slug="analytics",
    name="Text Analytics Agent",
    description=(
        "Analyses text: statistics, keyword extraction and PII redaction. "
        "Implemented with Google ADK and exposed over A2A."
    ),
    framework="google-adk",
    skills=[
        AgentSkill(
            id="text-statistics",
            name="Text statistics",
            description="Word, sentence and reading-time statistics for a passage.",
            tags=["text", "analytics"],
            examples=["How long would it take to read this paragraph? ..."],
        ),
        AgentSkill(
            id="keyword-extraction",
            name="Keyword extraction",
            description="Most frequent meaningful terms in a passage.",
            tags=["text", "nlp"],
            examples=["Give me the top 5 keywords of: ..."],
        ),
        AgentSkill(
            id="pii-redaction",
            name="PII redaction",
            description="Masks e-mail addresses and phone numbers.",
            tags=["privacy", "compliance"],
            examples=["Redact personal data from: contact me at jane@example.com"],
        ),
    ],
)

INSTRUCTION = (
    "You are a text analytics assistant. Always use the provided tools for statistics, "
    "keywords and redaction instead of estimating. Report tool results accurately and concisely. "
    "If the user supplies no text to analyse, ask for it."
)


def build_analytics_agent(model: BaseLlm | str) -> LlmAgent:
    return LlmAgent(
        name=APP_NAME,
        model=model,
        description=ANALYTICS_SPEC.description,
        instruction=INSTRUCTION,
        tools=[text_statistics, extract_keywords, redact_pii],
    )


def build_analytics_executor(model: BaseLlm | str) -> A2aAgentExecutor:
    runner = Runner(
        app_name=APP_NAME,
        agent=build_analytics_agent(model),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        auto_create_session=True,
    )
    return A2aAgentExecutor(runner=runner)


def build_agent(settings: Settings) -> tuple[AgentSpec, Any]:
    endpoint = resolve_endpoint(settings)
    model = LiteLlm(
        model=endpoint.litellm_model, api_base=endpoint.base_url, api_key=endpoint.api_key
    )
    return ANALYTICS_SPEC, build_analytics_executor(model)
