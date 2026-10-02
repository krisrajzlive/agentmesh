"""Ollama provider (local daemon or Ollama Cloud) using the native chat API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentmesh.llm.base import ChatMessage, LLMError, LLMResponse, LLMUnavailableError, Usage
from agentmesh.llm.http import HttpLLMProvider, raise_for_status


class OllamaProvider(HttpLLMProvider):
    name = "ollama"

    @staticmethod
    def _body(
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        *,
        stream: bool,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        return body

    async def _complete_once(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        response = await self._post(
            "/api/chat",
            self._body(
                self.model, messages, temperature, max_tokens, stream=False, json_mode=json_mode
            ),
        )
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("ollama: malformed response", provider=self.name) from exc
        return LLMResponse(
            content=content,
            provider=self.name,
            model=data.get("model", self.model),
            usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        )

    async def _stream_once(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int | None
    ) -> AsyncIterator[str]:
        body = self._body(self.model, messages, temperature, max_tokens, stream=True)
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=body, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                raise_for_status(response, self.name)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    if text := chunk.get("message", {}).get("content"):
                        yield text
                    if chunk.get("done"):
                        return
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("ollama: timeout", provider=self.name) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(
                f"ollama: connection error: {exc}", provider=self.name
            ) from exc
