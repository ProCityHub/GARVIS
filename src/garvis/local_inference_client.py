"""Persistent local inference transport for GARVIS.

The local model is a reasoning organ, never an authority source.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocalCompletion:
    content: str
    prompt_tps: float | None = None
    generation_tps: float | None = None
    prompt_ms: float | None = None
    generation_ms: float | None = None


def completion_payload(
    prompt: str,
    *,
    n_predict: int,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": max(1, int(n_predict)),
        "temperature": 0.0,
        "cache_prompt": True,
        "timings": True,
    }


def request_completion(
    server_url: str,
    prompt: str,
    *,
    n_predict: int,
    timeout_seconds: float,
) -> LocalCompletion:
    base = server_url.strip().rstrip("/")

    if not base:
        raise ValueError("local server URL must not be empty")

    body = json.dumps(
        completion_payload(
            prompt,
            n_predict=n_predict,
        )
    ).encode("utf-8")

    request = urllib.request.Request(
        base + "/completion",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=float(timeout_seconds),
        ) as response:
            raw = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"persistent local inference failed: {exc}"
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "persistent local inference returned invalid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "persistent local inference response must be an object"
        )

    content = payload.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "persistent local inference returned no content"
        )

    timings = payload.get("timings")
    if not isinstance(timings, dict):
        timings = {}

    def number(name: str) -> float | None:
        value = timings.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    return LocalCompletion(
        content=content,
        prompt_tps=number("prompt_per_second"),
        generation_tps=number("predicted_per_second"),
        prompt_ms=number("prompt_ms"),
        generation_ms=number("predicted_ms"),
    )
