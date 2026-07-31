import json
from dataclasses import replace

from garvis.health_aware_router import (
    ProviderHealthSnapshot,
    rank_remote_candidates,
    select_unblocked,
)
from garvis.universal_ai_registry import (
    AdapterKind,
    Authority,
    CandidateType,
    UniversalAIRegistry,
    android_app_organ,
    build_registry,
    garvis_evidence_organ,
    garvis_local_organ,
    hypercube_provider_plan,
    identify_provider,
    remote_model_organ,
)


def _multi_registry():
    return build_registry(
        [
            "gpt-test",
            "anthropic/claude-test",
            "grok/test",
            "groq/test",
            "openrouter/test",
            "compatible/test",
            "gemini/test",
        ],
        env={
            "OPENAI_API_KEY": "secret-openai",
            "ANTHROPIC_API_KEY": "secret-anthropic",
            "XAI_API_KEY": "secret-xai",
            "GROQ_API_KEY": "secret-groq",
            "OPENROUTER_API_KEY": "secret-openrouter",
            "GARVIS_COMPAT_API_KEY": "secret-compat",
            "GARVIS_COMPAT_BASE_URL": "https://example.invalid",
            "GEMINI_API_KEY": "secret-gemini",
        },
        capability_overrides={
            "gpt-test": {"text", "reasoning", "coding"},
            "anthropic/claude-test": {"text", "reasoning", "coding"},
            "grok/test": {"text", "reasoning", "research"},
            "groq/test": {"text"},
            "openrouter/test": {"text"},
            "compatible/test": {"text"},
            "gemini/test": {"text", "vision"},
        },
    )


def test_provider_families_resolve_explicitly():
    assert identify_provider("gpt-test").provider_id == "openai"
    assert identify_provider("anthropic/claude-test").provider_id == "anthropic"
    assert identify_provider("grok/test").provider_id == "xai"
    assert identify_provider("groq/test").provider_id == "groq"
    assert identify_provider("openrouter/test").provider_id == "openrouter"
    assert identify_provider("compatible/test").provider_id == "compatible"


def test_gemini_discovery_does_not_fake_adapter():
    identity = identify_provider("gemini/test")
    assert identity.provider_id == "gemini"
    assert identity.adapter is AdapterKind.MISSING
    assert identity.adapter_supported is False


def test_unconfigured_remote_never_becomes_programmable():
    organ = remote_model_organ("gpt-test", env={})
    assert organ.configured is False
    assert organ.programmable is False


def test_installed_app_does_not_imply_control():
    organ = android_app_organ("com.example.ai")
    assert organ.candidate_type is CandidateType.ANDROID_APP
    assert organ.programmable is False
    assert organ.adapter_supported is False
    assert organ.adapter is AdapterKind.MANUAL_ONLY


def test_external_provider_authority_is_candidate_only():
    registry = _multi_registry()
    remotes = [o for o in registry.all() if o.candidate_type is CandidateType.REMOTE_API]
    assert remotes
    assert all(o.authority is Authority.CANDIDATE_ONLY for o in remotes)


def test_garvis_context_evidence_integration_ownership_is_invariant():
    registry = _multi_registry()
    plan = {p.code: p for p in hypercube_provider_plan(registry)}
    assert plan["001"].owner == "garvis:local"
    assert plan["100"].owner == "garvis:evidence"
    assert plan["111"].owner == "garvis:local"


def test_hypercube_plan_has_exactly_eight_unique_perspectives():
    registry = _multi_registry()
    plan = hypercube_provider_plan(registry)
    assert len(plan) == 8
    assert {p.code for p in plan} == {
        "000", "001", "010", "011", "100", "101", "110", "111"
    }


def test_missing_adapter_is_not_routable_even_with_key_present():
    registry = _multi_registry()
    text = registry.candidates("text")
    assert all(item.provider_id != "gemini" for item in text)


def test_safe_report_never_contains_supplied_secret_values():
    registry = _multi_registry()
    encoded = json.dumps(registry.safe_report(), sort_keys=True)
    for secret in (
        "secret-openai",
        "secret-anthropic",
        "secret-xai",
        "secret-groq",
        "secret-openrouter",
        "secret-compat",
        "secret-gemini",
    ):
        assert secret not in encoded


def test_capability_filter_blocks_wrong_capability():
    registry = _multi_registry()
    coding = registry.candidates("coding")
    assert coding
    assert {o.provider_id for o in coding} <= {"openai", "anthropic"}


def test_health_order_is_not_authority_escalation():
    registry = _multi_registry()
    health = {
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test",
            last_success_at=100.0,
        ),
    }
    ranked = rank_remote_candidates(registry, health)
    assert ranked
    selected_ids = {item.organ_id for item in ranked}
    organs = {o.organ_id: o for o in registry.all()}
    assert all(organs[item].authority is Authority.CANDIDATE_ONLY for item in selected_ids)


def test_blocked_provider_is_never_selected():
    registry = _multi_registry()
    health = {
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test",
            blocked=True,
            last_success_at=100.0,
        )
    }
    selected = select_unblocked(rank_remote_candidates(registry, health), limit=99)
    assert all(item.model != "anthropic/claude-test" for item in selected)


def test_all_blocked_produces_empty_selection():
    registry = build_registry(
        ["gpt-test", "anthropic/claude-test"],
        env={"OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "y"},
    )
    health = {
        "gpt-test": ProviderHealthSnapshot("gpt-test", blocked=True),
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test", blocked=True
        ),
    }
    assert select_unblocked(rank_remote_candidates(registry, health), limit=10) == ()


def test_health_ranking_deterministic_under_input_reordering():
    env = {
        "OPENAI_API_KEY": "x",
        "ANTHROPIC_API_KEY": "y",
        "XAI_API_KEY": "z",
    }
    a = build_registry(
        ["gpt-test", "anthropic/claude-test", "grok/test"], env=env
    )
    b = build_registry(
        ["grok/test", "gpt-test", "anthropic/claude-test"], env=env
    )
    health = {
        "gpt-test": ProviderHealthSnapshot("gpt-test", failure_count=1),
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test", failure_count=0
        ),
        "grok/test": ProviderHealthSnapshot("grok/test", failure_count=2),
    }
    assert [x.model for x in rank_remote_candidates(a, health)] == [
        x.model for x in rank_remote_candidates(b, health)
    ]


def test_false_consensus_cannot_change_registry_authority():
    registry = _multi_registry()
    remotes = [
        replace(o, notes=o.notes + ("three providers agree",))
        for o in registry.all()
        if o.candidate_type is CandidateType.REMOTE_API
    ]
    assert remotes
    assert all(o.authority is Authority.CANDIDATE_ONLY for o in remotes)


def test_evidence_organ_is_not_provider_candidate():
    evidence = garvis_evidence_organ()
    assert evidence.authority is Authority.EVIDENCE_ONLY
    assert evidence.candidate_type is CandidateType.GARVIS_ORGAN


def test_local_garvis_retains_garvis_authority():
    local = garvis_local_organ()
    assert local.authority is Authority.GARVIS


def test_registry_replacement_by_same_id_is_deterministic():
    registry = UniversalAIRegistry()
    a = garvis_local_organ()
    registry.register(a)
    registry.register(a)
    assert len(registry.all()) == 1
