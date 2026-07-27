from garvis.health_aware_router import (
    ProviderHealthSnapshot,
    rank_remote_candidates,
    select_unblocked,
)
from garvis.universal_ai_registry import build_registry


def _registry():
    return build_registry(
        ["gpt-test", "anthropic/claude-test", "grok/test"],
        env={
            "OPENAI_API_KEY": "present",
            "ANTHROPIC_API_KEY": "present",
            "XAI_API_KEY": "present",
        },
    )


def test_blocked_provider_is_ranked_after_unblocked():
    registry = _registry()
    health = {
        "gpt-test": ProviderHealthSnapshot("gpt-test", blocked=True),
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test",
            last_success_at=1.0,
        ),
    }
    ranked = rank_remote_candidates(registry, health)
    assert ranked[0].model == "anthropic/claude-test"
    assert ranked[-1].model == "gpt-test"


def test_recorded_success_precedes_unproven_when_both_unblocked():
    registry = _registry()
    health = {
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test",
            last_success_at=1.0,
        )
    }
    ranked = rank_remote_candidates(registry, health)
    assert ranked[0].model == "anthropic/claude-test"


def test_fewer_failures_breaks_unproven_tie():
    registry = _registry()
    health = {
        "gpt-test": ProviderHealthSnapshot("gpt-test", failure_count=2),
        "anthropic/claude-test": ProviderHealthSnapshot(
            "anthropic/claude-test",
            failure_count=1,
        ),
        "grok/test": ProviderHealthSnapshot("grok/test", failure_count=0),
    }
    ranked = rank_remote_candidates(registry, health)
    assert [item.model for item in ranked][:3] == [
        "grok/test",
        "anthropic/claude-test",
        "gpt-test",
    ]


def test_select_unblocked_excludes_blocked():
    registry = _registry()
    health = {
        "gpt-test": ProviderHealthSnapshot("gpt-test", blocked=True),
    }
    selected = select_unblocked(rank_remote_candidates(registry, health), limit=10)
    assert all(item.model != "gpt-test" for item in selected)


def test_reason_explicitly_preserves_verification_boundary():
    registry = _registry()
    ranked = rank_remote_candidates(registry, {})
    assert ranked
    assert "verification" in ranked[0].reason
