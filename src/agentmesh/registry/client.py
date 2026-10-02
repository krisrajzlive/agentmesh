"""Client for the registry API (used by the orchestrator and the CLI)."""

from __future__ import annotations

from typing import Any

import httpx

from agentmesh.registry.models import AgentRecord


class RegistryClientError(Exception):
    pass


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key
        if bearer_token:
            self._headers["authorization"] = f"Bearer {bearer_token}"

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(
                method, f"{self._base}{path}", headers=self._headers, **kwargs
            )
        except httpx.HTTPError as exc:
            raise RegistryClientError(f"registry unreachable: {exc}") from exc
        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise RegistryClientError(f"registry returned {response.status_code}: {message}")
        return response

    async def register(self, url: str) -> AgentRecord:
        return AgentRecord.model_validate(
            (await self._call("POST", "/v1/agents", json={"url": url})).json()
        )

    async def list(
        self,
        *,
        skill: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        status: str | None = None,
    ) -> list[AgentRecord]:
        params = {
            k: v for k, v in {"skill": skill, "tag": tag, "q": query, "status": status}.items() if v
        }
        data = (await self._call("GET", "/v1/agents", params=params)).json()
        return [AgentRecord.model_validate(item) for item in data["agents"]]

    async def get(self, agent_id: str) -> AgentRecord:
        return AgentRecord.model_validate(
            (await self._call("GET", f"/v1/agents/{agent_id}")).json()
        )

    async def deregister(self, agent_id: str) -> None:
        await self._call("DELETE", f"/v1/agents/{agent_id}")
