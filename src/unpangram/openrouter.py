"""Minimal OpenRouter client (OpenAI-compatible chat completions and embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class OpenRouterError(Exception):
    pass


@dataclass
class ChatResult:
    text: str
    usage: dict[str, Any]
    finish_reason: str | None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
            transport=transport,
        )

    def chat(
        self, model: str, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> ChatResult:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = self._post("/chat/completions", body)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"unexpected chat response from {model}: {data}") from exc
        return ChatResult(
            text=text, usage=data.get("usage") or {}, finish_reason=choice.get("finish_reason")
        )

    def embed(self, model: str, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        data = self._post("/embeddings", {"model": model, "input": inputs})
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise OpenRouterError(f"unexpected embeddings response from {model}: {data}") from exc
        if len(vectors) != len(inputs):
            raise OpenRouterError(f"expected {len(inputs)} embeddings, got {len(vectors)}")
        return vectors, data.get("usage") or {}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(path, json=body)
        if resp.is_error:
            raise OpenRouterError(
                f"OpenRouter {path} failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise OpenRouterError(f"OpenRouter {path} error: {data['error']}")
        return data
