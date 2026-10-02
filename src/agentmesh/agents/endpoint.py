"""Resolve which LLM endpoint framework-based agents (ADK, Agent Framework) should use.

Those frameworks bring their own model clients, so they cannot share AgentMesh's
:class:`~agentmesh.llm.LLMProvider` failover chain. Instead we point them at the
first configured provider, using each vendor's OpenAI-compatible surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentmesh.config import Settings


class AgentConfigurationError(RuntimeError):
    """The agent cannot start with the current configuration."""


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    provider: str
    model: str
    base_url: str  # OpenAI-compatible, ends in /v1
    api_key: str

    @property
    def litellm_model(self) -> str:
        """Model id for LiteLLM's generic OpenAI-compatible route."""
        return f"openai/{self.model}"


def resolve_endpoint(settings: Settings) -> ModelEndpoint:
    for name in settings.llm_providers:
        if name == "openai" and settings.openai_api_key:
            return ModelEndpoint(
                "openai",
                settings.openai_model,
                settings.openai_base_url,
                settings.openai_api_key.get_secret_value(),
            )
        if name == "huggingface" and settings.hf_token:
            return ModelEndpoint(
                "huggingface",
                settings.hf_model,
                settings.hf_base_url,
                settings.hf_token.get_secret_value(),
            )
        if name == "ollama":
            key = (
                settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else "ollama"
            )
            return ModelEndpoint(
                "ollama",
                settings.ollama_model,
                f"{settings.ollama_base_url.rstrip('/')}/v1",
                key,
            )
    raise AgentConfigurationError(
        "this agent needs an LLM: set AGENTMESH_LLM_PROVIDERS and the matching credentials "
        "(OPENAI_API_KEY, HF_TOKEN or OLLAMA_API_KEY)"
    )
