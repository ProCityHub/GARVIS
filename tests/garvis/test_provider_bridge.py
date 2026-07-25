"""Tests for the GARVIS remote provider bridge."""

from __future__ import annotations

import pytest

from garvis import provider_bridge
from garvis.cli import _check_configuration


def _capture(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    monkeypatch.setattr(provider_bridge, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(
        provider_bridge,
        "set_default_openai_client",
        lambda client, use_for_tracing: captured.update(
            client_object=client, use_for_tracing=use_for_tracing
        ),
    )
    monkeypatch.setattr(
        provider_bridge,
        "set_default_openai_api",
        lambda value: captured.update(api=value),
    )
    monkeypatch.setattr(
        provider_bridge,
        "set_tracing_disabled",
        lambda value: captured.update(tracing_disabled=value),
    )
    return captured


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("groq/openai/gpt-oss-20b", "groq"),
        ("grok/grok-4.5", "grok"),
        ("openrouter/openrouter/free", "openrouter"),
        ("compatible/vendor/model", "compatible"),
    ],
)
def test_provider_prefixes(model: str, provider: str) -> None:
    assert provider_bridge.is_openai_compatible_model(model) is True
    assert provider_bridge.provider_name(model) == provider


def test_unknown_prefix_is_not_claimed() -> None:
    assert provider_bridge.is_openai_compatible_model("gpt-5.1") is False


def test_groq_configures_existing_openai_sdk(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    captured = _capture(monkeypatch)
    model = provider_bridge.configure_openai_compatible("groq/openai/gpt-oss-20b")
    assert model == "openai/gpt-oss-20b"
    assert captured["client"]["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["client"]["api_key"] == "test-groq-key"
    assert captured["use_for_tracing"] is False
    assert captured["api"] == "chat_completions"
    assert captured["tracing_disabled"] is True


def test_openrouter_configures_existing_openai_sdk(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    captured = _capture(monkeypatch)
    model = provider_bridge.configure_openai_compatible("openrouter/openrouter/free")
    assert model == "openrouter/free"
    assert captured["client"]["base_url"] == "https://openrouter.ai/api/v1"


def test_generic_provider_requires_https(monkeypatch) -> None:
    monkeypatch.setenv("GARVIS_COMPAT_API_KEY", "test-key")
    monkeypatch.setenv("GARVIS_COMPAT_BASE_URL", "http://example.com/v1")
    with pytest.raises(provider_bridge.ProviderConfigurationError):
        provider_bridge.configure_openai_compatible("compatible/vendor/model")


def test_generic_provider_accepts_https(monkeypatch) -> None:
    monkeypatch.setenv("GARVIS_COMPAT_API_KEY", "test-key")
    monkeypatch.setenv("GARVIS_COMPAT_BASE_URL", "https://models.example/v1/")
    captured = _capture(monkeypatch)
    model = provider_bridge.configure_openai_compatible("compatible/vendor/model")
    assert model == "vendor/model"
    assert captured["client"]["base_url"] == "https://models.example/v1"


def test_cli_accepts_configured_groq(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _check_configuration("groq/openai/gpt-oss-20b") is None


def test_cli_rejects_unconfigured_openrouter(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    error = _check_configuration("openrouter/openrouter/free")
    assert error is not None
    assert "OPENROUTER_API_KEY" in error


def test_grok_configures_existing_openai_sdk(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    captured = _capture(monkeypatch)
    model = provider_bridge.configure_openai_compatible("grok/grok-4.5")
    assert model == "grok-4.5"
    assert captured["client"]["base_url"] == "https://api.x.ai/v1"
    assert captured["client"]["api_key"] == "test-xai-key"
    assert captured["use_for_tracing"] is False
    assert captured["api"] == "chat_completions"
    assert captured["tracing_disabled"] is True


def test_grok_and_groq_do_not_cross_match() -> None:
    """One letter apart, two different companies. They must not collide."""

    assert provider_bridge.provider_name("grok/grok-4.5") == "grok"
    assert provider_bridge.provider_name("groq/openai/gpt-oss-20b") == "groq"
    assert provider_bridge.required_api_key_environment("grok/grok-4.5") == "XAI_API_KEY"
    assert provider_bridge.required_api_key_environment("groq/openai/gpt-oss-20b") == "GROQ_API_KEY"


def test_grok_requires_a_model_identifier(monkeypatch) -> None:
    """An empty model id is still claimed by grok/, so the error is specific."""

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    assert provider_bridge.is_openai_compatible_model("grok/") is True
    with pytest.raises(provider_bridge.ProviderConfigurationError):
        provider_bridge.configure_openai_compatible("grok/")


def test_cli_rejects_unconfigured_grok(monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    error = _check_configuration("grok/grok-4.5")
    assert error is not None
    assert "XAI_API_KEY" in error


def test_cli_accepts_configured_grok(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _check_configuration("grok/grok-4.5") is None
