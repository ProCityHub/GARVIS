"""Tests for the THANOS MODE standing authorization."""

from __future__ import annotations

import json

import pytest

from garvis.thanos_mode import (
    DEFAULT_ALLOWED_ACTIONS,
    OWNER,
    REPOSITORY,
    MergePreconditions,
    ThanosAction,
    ThanosAuthorizationStore,
    ThanosNotEnabledError,
    ThanosPausedError,
    ThanosRevokedError,
    ThanosScopeError,
    ThanosTamperError,
    create_authorization,
    evaluate_merge_gate,
    is_protected_path,
    pause_authorization,
    permits,
    render_status,
    resume_authorization,
    revoke_authorization,
)


def test_creation_is_enabled_and_sealed() -> None:
    auth = create_authorization()
    assert auth.owner == OWNER
    assert auth.repository == REPOSITORY
    assert auth.enabled is True
    assert auth.paused is False
    assert auth.is_active is True
    assert auth.verify() is True
    assert auth.authorization_version == 1


def test_merge_is_in_default_allowed_actions() -> None:
    assert ThanosAction.MERGE in DEFAULT_ALLOWED_ACTIONS
    assert ThanosAction.MERGE.value in create_authorization().allowed_actions


def test_active_authorization_permits_merge() -> None:
    permits(create_authorization(), ThanosAction.MERGE)


def test_repeated_merge_checks_do_not_consume_authorization() -> None:
    auth = create_authorization()
    for _ in range(50):
        permits(auth, ThanosAction.MERGE)
    assert auth.is_active is True


def test_paused_authorization_blocks_merge() -> None:
    paused = pause_authorization(create_authorization())
    with pytest.raises(ThanosPausedError):
        permits(paused, ThanosAction.MERGE)


def test_revoked_authorization_blocks_merge() -> None:
    revoked = revoke_authorization(create_authorization(), reason="owner stop")
    with pytest.raises(ThanosRevokedError):
        permits(revoked, ThanosAction.MERGE)


def test_repository_mismatch_blocks_merge() -> None:
    with pytest.raises(ThanosScopeError):
        permits(create_authorization(), ThanosAction.MERGE, repository="ProCityHub/AGI")


def test_runtime_mismatch_blocks_merge() -> None:
    with pytest.raises(ThanosScopeError):
        permits(create_authorization(), ThanosAction.MERGE, runtime_scope="other")


def test_tampered_authorization_blocks_merge() -> None:
    from dataclasses import replace

    auth = create_authorization()
    tampered = replace(auth, owner="Someone Else")
    with pytest.raises(ThanosTamperError):
        permits(tampered, ThanosAction.MERGE)


def test_status_reports_autonomous_merge_when_green() -> None:
    status = render_status(create_authorization())
    assert "AUTONOMOUS_MERGE_WHEN_GREEN=ENABLED" in status
    assert "OWNER_MERGE_CHECKPOINTS_PER_CYCLE=0" in status


def test_authorization_is_not_consumed_by_use() -> None:
    auth = create_authorization()
    for _ in range(50):
        permits(auth, ThanosAction.COMMIT)
        permits(auth, ThanosAction.PUSH)
        permits(auth, ThanosAction.CREATE_PULL_REQUEST)
    assert auth.is_active is True


def test_zero_per_stage_prompts_in_status() -> None:
    assert "PER_STAGE_APPROVAL_PROMPTS=0" in render_status(create_authorization())


def test_repository_scope_is_enforced() -> None:
    auth = create_authorization()
    with pytest.raises(ThanosScopeError):
        permits(auth, ThanosAction.EDIT, repository="ProCityHub/AGI")


def test_runtime_scope_is_enforced() -> None:
    auth = create_authorization()
    with pytest.raises(ThanosScopeError):
        permits(auth, ThanosAction.EDIT, runtime_scope="other-runtime")


def test_pause_and_resume() -> None:
    auth = create_authorization()
    paused = pause_authorization(auth)
    assert paused.is_active is False
    with pytest.raises(ThanosPausedError):
        permits(paused, ThanosAction.EDIT)
    resumed = resume_authorization(paused)
    assert resumed.is_active is True
    permits(resumed, ThanosAction.EDIT)


def test_revocation_stops_future_work() -> None:
    auth = create_authorization()
    revoked = revoke_authorization(auth, reason="owner stopped the experiment")
    assert revoked.is_revoked is True
    assert revoked.enabled is False
    with pytest.raises(ThanosRevokedError):
        permits(revoked, ThanosAction.EDIT)


def test_revocation_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        revoke_authorization(create_authorization(), reason="   ")


def test_revoked_grant_cannot_be_resumed() -> None:
    revoked = revoke_authorization(create_authorization(), reason="stop")
    with pytest.raises(ThanosRevokedError):
        resume_authorization(revoked)


def test_disabled_grant_raises_not_enabled() -> None:
    from dataclasses import replace

    auth = create_authorization()
    disabled = replace(auth, enabled=False, record_hash="").sealed()
    with pytest.raises(ThanosNotEnabledError):
        permits(disabled, ThanosAction.EDIT)


def test_unknown_action_is_rejected() -> None:
    auth = create_authorization()
    with pytest.raises(ThanosScopeError):
        permits(auth, "transfer-money")


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_persistence_round_trip(tmp_path) -> None:
    store = ThanosAuthorizationStore(tmp_path / "thanos.json")
    assert store.load() is None
    auth = create_authorization()
    store.append(auth)

    reloaded = ThanosAuthorizationStore(tmp_path / "thanos.json").load()
    assert reloaded is not None
    assert reloaded.authorization_id == auth.authorization_id
    assert reloaded.record_hash == auth.record_hash
    assert reloaded.is_active is True


def test_chain_head_reflects_latest_amendment(tmp_path) -> None:
    store = ThanosAuthorizationStore(tmp_path / "thanos.json")
    auth = store.append(create_authorization())
    store.append(pause_authorization(auth))

    head = store.load()
    assert head is not None
    assert head.paused is True
    assert len(store.history()) == 2


def test_revocation_survives_restart(tmp_path) -> None:
    path = tmp_path / "thanos.json"
    store = ThanosAuthorizationStore(path)
    auth = store.append(create_authorization())
    store.append(revoke_authorization(auth, reason="owner revoked"))

    fresh = ThanosAuthorizationStore(path).load()
    assert fresh is not None
    with pytest.raises(ThanosRevokedError):
        permits(fresh, ThanosAction.COMMIT)


def test_broken_link_is_rejected(tmp_path) -> None:
    store = ThanosAuthorizationStore(tmp_path / "thanos.json")
    store.append(create_authorization())
    orphan = create_authorization()  # links to genesis, not to the head
    with pytest.raises(ThanosTamperError):
        store.append(orphan)


def test_tampered_record_is_detected(tmp_path) -> None:
    path = tmp_path / "thanos.json"
    store = ThanosAuthorizationStore(path)
    store.append(create_authorization())

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chain"][0]["allowed_actions"].append("merge")
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ThanosTamperError):
        ThanosAuthorizationStore(path).load()


def test_revocation_cannot_be_erased_silently(tmp_path) -> None:
    path = tmp_path / "thanos.json"
    store = ThanosAuthorizationStore(path)
    auth = store.append(create_authorization())
    store.append(revoke_authorization(auth, reason="owner revoked"))

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chain"][1]["revoked_at"] = None
    raw["chain"][1]["enabled"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ThanosTamperError):
        ThanosAuthorizationStore(path).load()


def test_corrupt_store_is_detected(tmp_path) -> None:
    path = tmp_path / "thanos.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThanosTamperError):
        ThanosAuthorizationStore(path).load()


def test_unsealed_record_is_not_persisted(tmp_path) -> None:
    from dataclasses import replace

    store = ThanosAuthorizationStore(tmp_path / "thanos.json")
    auth = replace(create_authorization(), record_hash="")
    with pytest.raises(ThanosTamperError):
        store.append(auth)


def test_atomic_write_leaves_no_temp_files(tmp_path) -> None:
    store = ThanosAuthorizationStore(tmp_path / "thanos.json")
    store.append(create_authorization())
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".thanos-")]
    assert leftovers == []


# --------------------------------------------------------------------------
# Protected paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/garvis/thanos_mode.py",
        "src/garvis/stage_gate.py",
        "src/garvis/github_maintenance.py",
        ".github/workflows/tests.yml",
        ".github/CODEOWNERS",
    ],
)
def test_governance_paths_are_protected(path: str) -> None:
    assert is_protected_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["src/garvis/assistant.py", "docs/README.md", "tests/garvis/test_cli.py"],
)
def test_ordinary_paths_are_not_protected(path: str) -> None:
    assert is_protected_path(path) is False


def test_status_for_absent_authorization() -> None:
    status = render_status(None)
    assert "THANOS_MODE=DISABLED" in status
    assert "STANDING_AUTHORITY=ABSENT" in status


# --------------------------------------------------------------------------
# Autonomous merge gate
# --------------------------------------------------------------------------


def _green(
    *,
    repository: str = REPOSITORY,
    changed_paths: tuple[str, ...] = ("src/garvis/assistant.py",),
    expected_head_sha: str = "abc123",
    actual_head_sha: str = "abc123",
    required_checks_complete: bool = True,
    required_checks_passed: bool = True,
    tested_artifact_sha: str = "tree1",
    proposed_artifact_sha: str = "tree1",
    base_sha_at_test_time: str = "base1",
    base_sha_now: str = "base1",
    secrets_detected: bool = False,
    rollback_available: bool = True,
) -> MergePreconditions:
    return MergePreconditions(
        repository=repository,
        changed_paths=changed_paths,
        expected_head_sha=expected_head_sha,
        actual_head_sha=actual_head_sha,
        required_checks_complete=required_checks_complete,
        required_checks_passed=required_checks_passed,
        tested_artifact_sha=tested_artifact_sha,
        proposed_artifact_sha=proposed_artifact_sha,
        base_sha_at_test_time=base_sha_at_test_time,
        base_sha_now=base_sha_now,
        secrets_detected=secrets_detected,
        rollback_available=rollback_available,
    )


def test_green_ordinary_change_merges_autonomously() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green())
    assert decision.allowed is True
    assert decision.blocking_reasons == ()
    assert decision.requires_owner_review is False


def test_moved_head_blocks_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(actual_head_sha="def456"))
    assert decision.allowed is False
    assert any("head moved" in r for r in decision.blocking_reasons)


def test_failed_checks_block_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(required_checks_passed=False))
    assert decision.allowed is False


def test_incomplete_checks_block_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(required_checks_complete=False))
    assert decision.allowed is False


def test_artifact_mismatch_blocks_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(proposed_artifact_sha="tree2"))
    assert decision.allowed is False


def test_advanced_base_blocks_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(base_sha_now="base2"))
    assert decision.allowed is False
    assert any("unrelated work" in r for r in decision.blocking_reasons)


def test_detected_secret_blocks_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(secrets_detected=True))
    assert decision.allowed is False


def test_missing_rollback_blocks_merge() -> None:
    decision = evaluate_merge_gate(create_authorization(), _green(rollback_available=False))
    assert decision.allowed is False


def test_revoked_authorization_blocks_merge_gate() -> None:
    revoked = revoke_authorization(create_authorization(), reason="owner stop")
    assert evaluate_merge_gate(revoked, _green()).allowed is False


def test_governance_change_requires_owner_review() -> None:
    decision = evaluate_merge_gate(
        create_authorization(),
        _green(changed_paths=("src/garvis/assistant.py", "src/garvis/thanos_mode.py")),
    )
    assert decision.allowed is False
    assert decision.requires_owner_review is True
    assert "src/garvis/thanos_mode.py" in decision.protected_paths


def test_workflow_change_requires_owner_review() -> None:
    decision = evaluate_merge_gate(
        create_authorization(), _green(changed_paths=(".github/workflows/tests.yml",))
    )
    assert decision.requires_owner_review is True
