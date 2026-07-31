"""Tests for the Anthropic backend (Claude via OpenAI-compatible endpoint)."""

from typing import Any

import pytest

import garvis.anthropic_backend as ab
from garvis.anthropic_backend import (
    ANTHROPIC_BASE_URL,
    AnthropicConfigurationError,
    configure_anthropic,
    is_anthropic_model,
)


def test_model_detection():
    assert is_anthropic_model("anthropic/claude-sonnet-4-6") is True
    assert is_anthropic_model("claude-sonnet-4-6") is True
    assert is_anthropic_model("Claude-Opus-4-8") is True
    assert is_anthropic_model("gpt-5.1") is False
    assert is_anthropic_model("litellm/anthropic/claude-sonnet-4-6") is False
    assert is_anthropic_model(None) is False


def test_configure_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AnthropicConfigurationError, match="ANTHROPIC_API_KEY"):
        configure_anthropic("anthropic/claude-sonnet-4-6")


def test_configure_points_client_at_anthropic(monkeypatch):
    calls: dict[str, Any] = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        ab, "set_default_openai_client",
        lambda client, use_for_tracing=True: calls.update(
            base_url=str(client.base_url), tracing=use_for_tracing))
    monkeypatch.setattr(
        ab, "set_default_openai_api", lambda api: calls.update(api=api))
    monkeypatch.setattr(
        ab, "set_tracing_disabled", lambda flag: calls.update(traceoff=flag))

    bare = configure_anthropic("anthropic/claude-sonnet-4-6")
    assert bare == "claude-sonnet-4-6"
    assert calls["base_url"].startswith(ANTHROPIC_BASE_URL.rstrip("/"))
    assert calls["tracing"] is False
    assert calls["api"] == "chat_completions"
    assert calls["traceoff"] is True


def test_configure_keeps_bare_claude_name(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(ab, "set_default_openai_client", lambda *a, **k: None)
    monkeypatch.setattr(ab, "set_default_openai_api", lambda *a, **k: None)
    monkeypatch.setattr(ab, "set_tracing_disabled", lambda *a, **k: None)
    assert configure_anthropic("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_assistant_routes_anthropic_model(monkeypatch):
    import garvis.assistant as assistant_mod
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    seen = {}

    def fake_configure(model):
        seen["model"] = model
        return "claude-sonnet-4-6"

    monkeypatch.setattr(assistant_mod, "configure_anthropic", fake_configure)
    a = assistant_mod.GarvisAssistant(
        model="anthropic/claude-sonnet-4-6", persist_memory=False)
    assert a.model == "claude-sonnet-4-6"
    assert seen["model"] == "anthropic/claude-sonnet-4-6"


def test_assistant_openai_models_untouched(monkeypatch):
    import garvis.assistant as assistant_mod

    def boom(model):
        raise AssertionError("must not configure anthropic for openai models")

    monkeypatch.setattr(assistant_mod, "configure_anthropic", boom)
    a = assistant_mod.GarvisAssistant(model="gpt-5.1", persist_memory=False)
    assert a.model == "gpt-5.1"


def test_cli_check_configuration_anthropic(monkeypatch):
    from garvis.cli import _check_configuration
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    err = _check_configuration("anthropic/claude-sonnet-4-6")
    assert err is not None and "ANTHROPIC_API_KEY" in err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert _check_configuration("anthropic/claude-sonnet-4-6") is None


def test_cli_check_configuration_openai_unchanged(monkeypatch):
    from garvis.cli import _check_configuration
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GARVIS_MODEL", raising=False)
    err = _check_configuration(None)
    assert err is not None and "OPENAI_API_KEY" in err
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    assert _check_configuration(None) is None


def test_cli_check_configuration_litellm_passthrough(monkeypatch):
    from garvis.cli import _check_configuration
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _check_configuration("litellm/anthropic/claude-sonnet-4-6") is None


def test_no_key_material_in_module_source():
    import inspect
    src = inspect.getsource(ab)
    assert "sk-ant" not in src and "sk-proj" not in src
