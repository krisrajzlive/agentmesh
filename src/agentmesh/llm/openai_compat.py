"""OpenAI-compatible chat-completions provider (OpenAI, Hugging Face router, vLLM, ...)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentmesh.llm.base import ChatMessage, LLMError, LLMResponse, LLMUnavailableError, Usage
from agentmesh.llm.http import HttpLLMProvider, raise_for_status


def _payload(
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int | None,
    *,
    json_mode: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream"] = True
    return body


class OpenAICompatibleProvider(HttpLLMProvider):
    def __init__(self, *, name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name

    async def _complete_once(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        response = await self._post(
            "/chat/completions",
            _payload(self.model, messages, temperature, max_tokens, json_mode=json_mode),
        )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: malformed response", provider=self.name) from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=data.get("model", self.model),
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        )

    async def _stream_once(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int | None
    ) -> AsyncIterator[str]:
        body = _payload(self.model, messages, temperature, max_tokens, stream=True)
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json=body, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                raise_for_status(response, self.name)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(f"{self.name}: timeout", provider=self.name) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(
                f"{self.name}: connection error: {exc}", provider=self.name
            ) from exc
