"""Anthropic (Claude) backend for GARVIS via the OpenAI-compatible endpoint.

Usage: set the model to "anthropic/<model>" (e.g. "anthropic/claude-sonnet-4-6")
or any name starting with "claude", via --model or GARVIS_MODEL, with
ANTHROPIC_API_KEY in the environment.

Governance (enforced by tests):
- The API key is read from the environment only; never logged or committed.
- Tracing upload is disabled in Anthropic mode.
- Configuration fails loudly when the key is absent.

Authorship: Adrien D. Thomas / ProCityHub.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"


class AnthropicConfigurationError(RuntimeError):
    """Raised when Anthropic mode is requested without valid configuration."""


def is_anthropic_model(model: object) -> bool:
    if not isinstance(model, str):
        return False
    name = model.strip().lower()
    return name.startswith("anthropic/") or name.startswith("claude")


def configure_anthropic(model: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not api_key.strip():
        raise AnthropicConfigurationError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment "
            "(e.g. ~/.secrets/anthropic.env, chmod 600, never committed) "
            "before running GARVIS on an Anthropic model."
        )
    client = AsyncOpenAI(base_url=ANTHROPIC_BASE_URL, api_key=api_key)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    bare = model.strip()
    if bare.lower().startswith("anthropic/"):
        bare = bare[len("anthropic/"):]
    return bare
