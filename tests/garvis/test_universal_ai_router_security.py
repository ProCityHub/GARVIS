import json

from garvis.universal_ai_registry import (
    AdapterKind,
    Authority,
    CandidateType,
    build_registry,
    identify_provider,
    remote_model_organ,
)


def test_unknown_provider_fails_closed():
    for model in (
        "claudeevil",
        "anthropicevil/model",
        "grokish/model",
        "totally-unknown-model",
        "vendor/model",
        "../model",
        "openai-ish/model",
    ):
        identity = identify_provider(model)
        assert identity.provider_id == "unknown"
        assert identity.adapter is AdapterKind.UNKNOWN
        assert identity.adapter_supported is False


def test_unknown_provider_is_never_programmable_even_with_openai_key():
    organ = remote_model_organ(
        "totally-unknown-model",
        env={"OPENAI_API_KEY": "secret"},
    )
    assert organ.provider_id == "unknown"
    assert organ.configured is False
    assert organ.programmable is False
    assert organ.adapter_supported is False
    assert organ.authority is Authority.CANDIDATE_ONLY


def test_explicit_openai_names_still_route_to_openai():
    for model in (
        "openai/gpt-test",
        "gpt-5.1",
        "gpt-4.1",
        "o1",
        "o3-mini",
        "o4-mini",
    ):
        identity = identify_provider(model)
        assert identity.provider_id == "openai"
        assert identity.adapter is AdapterKind.OPENAI_NATIVE
        assert identity.adapter_supported is True


def test_known_provider_families_still_route():
    expected = {
        "anthropic/claude-test": "anthropic",
        "claude-sonnet-test": "anthropic",
        "claude/test": "anthropic",
        "grok/test": "xai",
        "groq/test": "groq",
        "openrouter/test": "openrouter",
        "compatible/test": "compatible",
        "gemini/test": "gemini",
    }
    for model, provider in expected.items():
        assert identify_provider(model).provider_id == provider


def test_environment_presence_does_not_create_provider_identity():
    registry = build_registry(
        ["totally-unknown-model"],
        env={
            "OPENAI_API_KEY": "secret-openai",
            "ANTHROPIC_API_KEY": "secret-anthropic",
            "XAI_API_KEY": "secret-xai",
        },
    )
    remotes = [
        o for o in registry.all()
        if o.candidate_type is CandidateType.REMOTE_API
    ]
    assert len(remotes) == 1
    assert remotes[0].provider_id == "unknown"
    assert remotes[0].programmable is False


def test_safe_report_does_not_leak_configuration_values():
    registry = build_registry(
        ["openai/gpt-test", "totally-unknown-model"],
        env={"OPENAI_API_KEY": "very-secret-value"},
    )
    encoded = json.dumps(registry.safe_report(), sort_keys=True)
    assert "very-secret-value" not in encoded
