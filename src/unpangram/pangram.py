"""Pangram AI-detection client (async task API).

Docs: https://docs.pangram.com/api-reference/ai-detection
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

MIN_WORDS = 50
USD_PER_100_WORDS = 0.05
MAX_429_RETRIES = 5
DEFAULT_RETRY_AFTER = 5.0
MAX_RETRY_AFTER = 60.0
HUMAN_LABEL = "Human Written"


class PangramError(Exception):
    pass


def word_count(text: str) -> int:
    return len(text.split())


def pangram_cost(text: str) -> float:
    """USD for one scoring call: $0.05 per started 100 words."""
    return math.ceil(word_count(text) / 100) * USD_PER_100_WORDS


@dataclass
class Window:
    text: str
    label: str
    ai_assistance_score: float
    confidence: str
    start_index: int
    end_index: int
    is_humanized: bool
    humanizer_score: float

    @property
    def flagged(self) -> bool:
        return self.label != HUMAN_LABEL


@dataclass
class PangramResult:
    prediction_short: str
    fraction_ai: float
    fraction_ai_assisted: float
    fraction_human: float
    version: str
    windows: list[Window]
    raw: dict[str, Any]

    @property
    def flagged_windows(self) -> list[Window]:
        return [w for w in self.windows if w.flagged]

    @property
    def fully_human(self) -> bool:
        return self.prediction_short == "Human" and not self.flagged_windows


def parse_result(data: dict[str, Any]) -> PangramResult:
    windows = [
        Window(
            text=w.get("text", ""),
            label=w.get("label", ""),
            ai_assistance_score=float(w.get("ai_assistance_score") or 0.0),
            confidence=w.get("confidence", ""),
            start_index=int(w.get("start_index") or 0),
            end_index=int(w.get("end_index") or 0),
            is_humanized=bool(w.get("is_humanized", False)),
            humanizer_score=float(w.get("humanizer_score") or 0.0),
        )
        for w in data.get("windows") or []
    ]
    return PangramResult(
        prediction_short=data.get("prediction_short", ""),
        fraction_ai=float(data.get("fraction_ai") or 0.0),
        fraction_ai_assisted=float(data.get("fraction_ai_assisted") or 0.0),
        fraction_human=float(data.get("fraction_human") or 0.0),
        version=data.get("version", ""),
        windows=windows,
        raw=data,
    )


class PangramClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://text.external-api.pangram.com",
        model: str = "pangram-4",
        poll_interval: float = 1.0,
        poll_timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"x-api-key": api_key},
            timeout=30.0,
            transport=transport,
        )
        self._model = model
        self._poll_interval = poll_interval
        self._max_polls = max(1, math.ceil(poll_timeout / max(poll_interval, 0.001)))
        self._sleep = sleep

    def score(self, text: str) -> PangramResult:
        words = word_count(text)
        if words < MIN_WORDS:
            raise PangramError(f"text has {words} words; Pangram needs at least {MIN_WORDS}")
        created = self._request("POST", "/task", json={"text": text, "model": self._model})
        task_id = created.get("task_id")
        if not task_id:
            raise PangramError(f"Pangram returned no task_id: {created}")
        for _ in range(self._max_polls):
            data = self._request("GET", f"/task/{task_id}")
            stage = data.get("stage")
            if stage == "STAGE_SUCCESS":
                return parse_result(data)
            if stage == "STAGE_FAILED":
                raise PangramError(f"Pangram task failed: {data.get('headline') or 'no reason'}")
            self._sleep(self._poll_interval)
        raise PangramError(f"Pangram task {task_id} did not finish in time")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(MAX_429_RETRIES + 1):
            resp = self._http.request(method, path, **kwargs)
            if resp.status_code == 429 and attempt < MAX_429_RETRIES:
                self._sleep(_retry_after(resp))
                continue
            if resp.is_error:
                raise PangramError(
                    f"Pangram {method} {path} failed: HTTP {resp.status_code} {resp.text[:200]}"
                )
            return resp.json()
        raise AssertionError("unreachable")


def _retry_after(resp: httpx.Response) -> float:
    try:
        seconds = float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return DEFAULT_RETRY_AFTER
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)
