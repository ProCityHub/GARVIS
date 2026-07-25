"""Remote inference-provider bridge for GARVIS.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

GARVIS remains the orchestrator. Authorized external models are interchangeable
inference engines, not identities and not execution authorities. API keys are
read only from environment variables. Configuration performs no network call.

Python 3.9 compatible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from openai import AsyncOpenAI

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled

__all__ = [
    "ProviderConfigurationError",
    "ProviderSpec",
    "configure_openai_compatible",
    "is_openai_compatible_model",
    "provider_configuration_error",
    "provider_name",
    "required_api_key_environment",
]


class ProviderConfigurationError(RuntimeError):
    """Raised when a requested remote provider is not safely configured."""


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration required for one OpenAI-compatible provider."""

    name: str
    prefix: str
    base_url: str
    api_key_environment: str


_PROVIDER_SPECS = {
    "groq": ProviderSpec("groq", "groq/", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ProviderSpec(
        "openrouter", "openrouter/", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
    "compatible": ProviderSpec("compatible", "compatible/", "", "GARVIS_COMPAT_API_KEY"),
}


def _split_model(model: object) -> Optional[tuple[ProviderSpec, str]]:
    if not isinstance(model, str):
        return None
    clean = model.strip()
    lowered = clean.casefold()
    for spec in _PROVIDER_SPECS.values():
        if lowered.startswith(spec.prefix):
            bare_model = clean[len(spec.prefix) :].strip()
            if not bare_model:
                raise ProviderConfigurationError(f"{spec.name} model identifier is empty")
            return spec, bare_model
    return None


def is_openai_compatible_model(model: object) -> bool:
    try:
        return _split_model(model) is not None
    except ProviderConfigurationError:
        return True


def provider_name(model: object) -> Optional[str]:
    resolved = _split_model(model)
    return resolved[0].name if resolved is not None else None


def required_api_key_environment(model: object) -> Optional[str]:
    resolved = _split_model(model)
    return resolved[0].api_key_environment if resolved is not None else None


def _compatible_base_url(spec: ProviderSpec) -> str:
    if spec.name != "compatible":
        return spec.base_url
    base_url = os.getenv("GARVIS_COMPAT_BASE_URL", "").strip()
    if not base_url:
        raise ProviderConfigurationError("GARVIS_COMPAT_BASE_URL is not set for compatible/ models")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ProviderConfigurationError("GARVIS_COMPAT_BASE_URL must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise ProviderConfigurationError("GARVIS_COMPAT_BASE_URL must not contain credentials")
    return base_url.rstrip("/")


def _resolved_configuration(model: str) -> tuple[ProviderSpec, str, str, str]:
    resolved = _split_model(model)
    if resolved is None:
        raise ProviderConfigurationError(f"model is not handled by the provider bridge: {model}")
    spec, bare_model = resolved
    api_key = os.getenv(spec.api_key_environment, "").strip()
    if not api_key:
        raise ProviderConfigurationError(
            f"{spec.api_key_environment} is not set for {spec.name} provider"
        )
    return spec, bare_model, _compatible_base_url(spec), api_key


def provider_configuration_error(model: object) -> Optional[str]:
    if not isinstance(model, str):
        return None
    try:
        _resolved_configuration(model)
    except ProviderConfigurationError as error:
        return str(error)
    return None


def configure_openai_compatible(model: str) -> str:
    """Configure the Agents SDK and return the provider's bare model ID."""

    _spec, bare_model, base_url, api_key = _resolved_configuration(model)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    return bare_model
