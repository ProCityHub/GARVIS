"""GARVIS-owned FlexAI HTTPS inference adapter.

Creator / conceptual architecture: Adrien D. Thomas / ProCityHub.

Boundary:
- GARVIS retains routing, memory, governance, evidence, and tool authority.
- FlexAI is an interchangeable remote inference engine only.
- This module does not import OpenAI, Agents SDK, LangChain, LangGraph, or a
  FlexAI agent framework.
- Provider output is candidate information and never self-verifying truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import requests


FLEXAI_API_KEY_ENV = "FLEXAI_API_KEY"
FLEXAI_BASE_URL = "https://api.flex.ai/v1"
FLEXAI_CHAT_COMPLETIONS_URL = FLEXAI_BASE_URL + "/chat/completions"


class FlexAIConfigurationError(RuntimeError):
    """Raised before transport when FlexAI configuration is invalid."""


class FlexAITransportError(RuntimeError):
    """Raised when the remote transport fails without exposing response bodies."""


class FlexAIResponseError(RuntimeError):
    """Raised when FlexAI returns an unusable response shape."""


@dataclass(frozen=True)
class FlexAIResult:
    """Normalized candidate output plus preserved raw provider JSON."""

    model: str
    text: str
    raw: Mapping[str, Any]


def _bare_model(model: object) -> str:
    if not isinstance(model, str):
        raise FlexAIConfigurationError("FlexAI model identifier must be a string")

    clean = model.strip()
    prefix = "flexai/"
    if not clean.casefold().startswith(prefix):
        raise FlexAIConfigurationError(
            "FlexAI model identifier must use the explicit flexai/ namespace"
        )

    bare = clean[len(prefix):].strip()
    if not bare:
        raise FlexAIConfigurationError("FlexAI model identifier is empty")
    return bare


def is_flexai_model(model: object) -> bool:
    if not isinstance(model, str):
        return False
    return model.strip().casefold().startswith("flexai/")


def _resolve_api_key(env: Optional[Mapping[str, str]]) -> str:
    source = os.environ if env is None else env
    value = (source.get(FLEXAI_API_KEY_ENV) or "").strip()
    if not value:
        raise FlexAIConfigurationError(
            "FLEXAI_API_KEY is not set for the FlexAI provider"
        )
    return value


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise FlexAIConfigurationError("each message must be a mapping")
        normalized.append(dict(message))

    if not normalized:
        raise FlexAIConfigurationError("at least one message is required")
    return normalized


def chat_completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 30.0,
    post: Optional[Callable[..., Any]] = None,
) -> FlexAIResult:
    """Send one governed chat-completions request.

    ``post`` exists for deterministic offline testing. Production callers may
    omit it, in which case ``requests.post`` is used.
    """

    bare_model = _bare_model(model)
    api_key = _resolve_api_key(env)

    if timeout <= 0:
        raise FlexAIConfigurationError("timeout must be greater than zero")

    payload = {
        "model": bare_model,
        "messages": _normalize_messages(messages),
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    transport = requests.post if post is None else post

    try:
        response = transport(
            FLEXAI_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FlexAITransportError("FlexAI request transport failed") from exc

    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code < 200 or status_code >= 300:
        raise FlexAITransportError(
            "FlexAI returned HTTP status {}".format(status_code)
        )

    try:
        data = response.json()
    except (TypeError, ValueError) as exc:
        raise FlexAIResponseError("FlexAI response was not valid JSON") from exc

    if not isinstance(data, Mapping):
        raise FlexAIResponseError("FlexAI response root must be an object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise FlexAIResponseError("FlexAI response contained no choices")

    first = choices[0]
    if not isinstance(first, Mapping):
        raise FlexAIResponseError("FlexAI choice must be an object")

    message = first.get("message")
    if not isinstance(message, Mapping):
        raise FlexAIResponseError("FlexAI choice contained no message")

    content = message.get("content")
    if not isinstance(content, str):
        raise FlexAIResponseError("FlexAI message content must be text")

    return FlexAIResult(
        model=bare_model,
        text=content,
        raw=dict(data),
    )
