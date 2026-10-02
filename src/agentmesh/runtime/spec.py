"""Declarative agent description and Agent Card construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from google.protobuf.struct_pb2 import Struct

from agentmesh import __version__
from agentmesh.config import Settings

JSONRPC_PATH = "/a2a/jsonrpc"
REST_PATH = "/a2a/rest"
PROTOCOL_VERSION = "1.0"
RUNTIME_EXTENSION_URI = "urn:agentmesh:ext:runtime:v1"


@dataclass(frozen=True)
class AgentSpec:
    """Everything needed to publish an agent: identity, skills and capabilities."""

    slug: str
    name: str
    description: str
    skills: list[AgentSkill]
    version: str = __version__
    framework: str = "agentmesh"
    streaming: bool = True
    push_notifications: bool = True
    extensions: list[AgentExtension] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    extended_skills: list[AgentSkill] = field(default_factory=list)
    documentation_url: str = ""

    def __post_init__(self) -> None:
        if not self.skills:
            raise ValueError("an agent must declare at least one skill")


def _struct(values: dict[str, str]) -> Struct:
    struct = Struct()
    struct.update(values)
    return struct


def _security(settings: Settings) -> tuple[dict[str, SecurityScheme], list[SecurityRequirement]]:
    schemes: dict[str, SecurityScheme] = {}
    if settings.auth_mode in {"api_key", "api_key_or_jwt"}:
        schemes["apiKey"] = SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(
                location="header", name="X-API-Key", description="Static API key"
            )
        )
    if settings.auth_mode in {"jwt", "api_key_or_jwt"}:
        schemes["bearer"] = SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer", bearer_format="JWT")
        )
    # Any one of the declared schemes is sufficient (alternatives, not a conjunction).
    requirements = [SecurityRequirement(schemes={name: StringList()}) for name in schemes]
    return schemes, requirements


def build_agent_card(
    spec: AgentSpec,
    public_url: str,
    settings: Settings,
    *,
    extended: bool = False,
) -> AgentCard:
    """Build the public (or extended) Agent Card for ``spec`` served at ``public_url``."""
    base = public_url.rstrip("/")
    schemes, requirements = _security(settings)
    skills = [*spec.skills, *spec.extended_skills] if extended else list(spec.skills)
    return AgentCard(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        documentation_url=spec.documentation_url,
        provider=AgentProvider(organization="AgentMesh", url=base),
        supported_interfaces=[
            AgentInterface(
                url=f"{base}{JSONRPC_PATH}",
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION,
            ),
            AgentInterface(
                url=f"{base}{REST_PATH}",
                protocol_binding="HTTP+JSON",
                protocol_version=PROTOCOL_VERSION,
            ),
        ],
        capabilities=AgentCapabilities(
            streaming=spec.streaming,
            push_notifications=spec.push_notifications,
            extensions=[
                AgentExtension(
                    uri=RUNTIME_EXTENSION_URI,
                    description="Describes the framework and runtime hosting this agent.",
                    required=False,
                    params=_struct({"framework": spec.framework, "runtime_version": __version__}),
                ),
                *spec.extensions,
            ],
            extended_agent_card=bool(spec.extended_skills),
        ),
        security_schemes=schemes,
        security_requirements=requirements,
        default_input_modes=spec.input_modes,
        default_output_modes=spec.output_modes,
        skills=skills,
    )
