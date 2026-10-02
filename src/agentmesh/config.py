"""Typed, environment-driven configuration (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AuthMode = Literal["none", "api_key", "jwt", "api_key_or_jwt"]


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Process-wide settings. Every field is overridable via ``AGENTMESH_<NAME>``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENTMESH_",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # --- LLM providers -------------------------------------------------
    llm_providers: Annotated[list[str], NoDecode] = Field(default_factory=list)
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "AGENTMESH_OPENAI_API_KEY")
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    hf_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "AGENTMESH_HF_TOKEN")
    )
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OLLAMA_API_KEY", "AGENTMESH_OLLAMA_API_KEY")
    )
    ollama_model: str = "llama3.1"

    # --- Hosting --------------------------------------------------------
    task_store_url: str = "memory"
    auth_mode: AuthMode = "none"
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "agentmesh"
    jwt_issuer: str | None = None
    push_signing_secret: SecretStr | None = None
    allow_private_push_urls: bool = False

    @field_validator("llm_providers", "api_keys", mode="before")
    @classmethod
    def _csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("llm_providers")
    @classmethod
    def _known_providers(cls, value: list[str]) -> list[str]:
        known = {"openai", "huggingface", "ollama"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                f"unknown LLM provider(s) {unknown}; expected a subset of {sorted(known)}"
            )
        return value

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.environment == "production" and self.auth_mode == "none":
            raise ValueError("AGENTMESH_AUTH_MODE must not be 'none' in production")
        if self.auth_mode == "api_key" and not self.api_keys:
            raise ValueError("AGENTMESH_API_KEYS is required when auth_mode='api_key'")
        if self.auth_mode == "jwt" and self.jwt_secret is None:
            raise ValueError("AGENTMESH_JWT_SECRET is required when auth_mode='jwt'")
        if self.auth_mode == "api_key_or_jwt" and not (self.api_keys or self.jwt_secret):
            raise ValueError(
                "auth_mode='api_key_or_jwt' needs AGENTMESH_API_KEYS or AGENTMESH_JWT_SECRET"
            )
        return self

    def api_key_map(self) -> dict[str, str]:
        """Return ``{key: principal}`` parsed from ``name:key`` pairs."""
        mapping: dict[str, str] = {}
        for entry in self.api_keys:
            principal, sep, key = entry.partition(":")
            if not sep or not key:
                raise ValueError("AGENTMESH_API_KEYS entries must look like 'name:key'")
            mapping[key] = principal
        return mapping


@lru_cache
def get_settings() -> Settings:
    return Settings()
