"""Registry domain model."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from a2a.types import AgentCard
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, Field

from agentmesh.runtime.spec import RUNTIME_EXTENSION_URI


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class SkillInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentRecord(BaseModel):
    id: str
    name: str
    url: str  # base URL the agent was registered with
    description: str = ""
    version: str = ""
    framework: str = "unknown"
    skills: list[SkillInfo] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)
    streaming: bool = False
    push_notifications: bool = False
    requires_auth: bool = False
    status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_error: str | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_checked_at: datetime | None = None
    card: dict[str, Any] = Field(default_factory=dict)

    def matches(self, *, skill: str | None, tag: str | None, query: str | None) -> bool:
        if skill and all(s.id != skill for s in self.skills):
            return False
        if tag and all(tag not in s.tags for s in self.skills):
            return False
        if query:
            haystack = " ".join(
                [self.name, self.description, *(f"{s.name} {s.description}" for s in self.skills)]
            ).lower()
            if not all(word in haystack for word in query.lower().split()):
                return False
        return True


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "agent"


def record_from_card(url: str, card: AgentCard, *, agent_id: str | None = None) -> AgentRecord:
    """Project an Agent Card into a registry record."""
    framework = "unknown"
    for extension in card.capabilities.extensions:
        if extension.uri == RUNTIME_EXTENSION_URI and "framework" in extension.params:
            framework = str(extension.params["framework"])
    return AgentRecord(
        id=agent_id or slugify(card.name),
        name=card.name,
        url=url.rstrip("/"),
        description=card.description,
        version=card.version,
        framework=framework,
        skills=[
            SkillInfo(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=list(s.tags),
                examples=list(s.examples),
            )
            for s in card.skills
        ],
        input_modes=list(card.default_input_modes),
        output_modes=list(card.default_output_modes),
        streaming=card.capabilities.streaming,
        push_notifications=card.capabilities.push_notifications,
        requires_auth=bool(card.security_requirements),
        card=MessageToDict(card),
    )
