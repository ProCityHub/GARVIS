"""Tests for the THANOS autonomous upgrade-cycle state machine."""

from __future__ import annotations

import os

import pytest

from garvis.thanos_mode import (
    MergePreconditions,
    ThanosAction,
    create_authorization,
    evaluate_merge_gate,
    pause_authorization,
    revoke_authorization,
)
from garvis.upgrade_cycle import (
    CycleLock,
    CycleLockError,
    CycleState,
    CycleStore,
    CycleTransitionError,
    advance,
    new_cycle,
    record_merge_decision,
    transition_is_legal,
)


@pytest.fixture()
def auth():
    return create_authorization()


@pytest.fixture()
def cycle():
    return new_cycle(
        objective="repair failing lint job",
        starting_commit="049fc9d",
        starting_version="0.3.1",
    )


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


def test_pipeline_order_is_enforced(cycle, auth) -> None:
    moved = advance(cycle, CycleState.OBSERVING, auth)
    assert moved.state is CycleState.OBSERVING
    with pytest.raises(CycleTransitionError):
        advance(moved, CycleState.MERGING, auth)


def test_every_transition_is_recorded(cycle, auth) -> None:
    current = cycle
    for state in (
        CycleState.OBSERVING,
        CycleState.RESEARCHING,
        CycleState.DIAGNOSING,
    ):
        current = advance(current, state, auth)
    assert len(current.transitions) == 3
    assert current.transitions[0].source == CycleState.IDLE.value
    assert current.transitions[-1].destination == CycleState.DIAGNOSING.value
    assert all(t.occurred_at for t in current.transitions)


def test_failed_validation_returns_to_patching(cycle, auth) -> None:
    assert transition_is_legal(CycleState.TESTING, CycleState.PATCHING)
    assert transition_is_legal(CycleState.LINTING, CycleState.PATCHING)
    assert transition_is_legal(CycleState.SECURITY_REVIEWING, CycleState.PATCHING)


def test_ci_failure_enters_repair_loop(auth) -> None:
    assert transition_is_legal(CycleState.WAITING_FOR_CI, CycleState.REPAIRING_CI)
    assert transition_is_legal(CycleState.REPAIRING_CI, CycleState.PATCHING)
    assert transition_is_legal(CycleState.REPAIRING_CI, CycleState.WAITING_FOR_CI)


def test_unhealthy_candidate_can_roll_back() -> None:
    assert transition_is_legal(CycleState.HEALTH_CHECKING, CycleState.ROLLING_BACK)
    assert transition_is_legal(CycleState.PROMOTING, CycleState.ROLLING_BACK)
    assert transition_is_legal(CycleState.ROLLING_BACK, CycleState.OBSERVING)


def test_terminal_cycle_cannot_advance(cycle, auth) -> None:
    failed = advance(cycle, CycleState.FAILED, auth)
    assert failed.is_terminal is True
    with pytest.raises(CycleTransitionError):
        advance(failed, CycleState.OBSERVING, auth)


def test_no_per_stage_prompt_is_emitted(cycle, auth, capsys) -> None:
    current = cycle
    for state in (
        CycleState.OBSERVING,
        CycleState.RESEARCHING,
        CycleState.DIAGNOSING,
        CycleState.SPECIFYING,
        CycleState.PLANNING,
    ):
        current = advance(current, state, auth)
    assert capsys.readouterr().out == ""
    assert current.state is CycleState.PLANNING


# --------------------------------------------------------------------------
# Authorization interaction
# --------------------------------------------------------------------------


def test_revoked_authorization_blocks_the_cycle(cycle) -> None:
    revoked = revoke_authorization(create_authorization(), reason="owner stop")
    blocked = advance(cycle, CycleState.OBSERVING, revoked)
    assert blocked.state is CycleState.BLOCKED
    assert "revoked" in (blocked.blocker or "").lower()


def test_paused_authorization_blocks_the_cycle(cycle) -> None:
    paused = pause_authorization(create_authorization())
    blocked = advance(cycle, CycleState.OBSERVING, paused)
    assert blocked.state is CycleState.BLOCKED


def test_owner_pause_is_always_reachable(cycle, auth) -> None:
    working = advance(cycle, CycleState.OBSERVING, auth)
    paused = advance(working, CycleState.PAUSED, auth)
    assert paused.state is CycleState.PAUSED
    assert transition_is_legal(CycleState.PAUSED, CycleState.OBSERVING)


def test_authorization_is_not_consumed_across_a_full_pipeline(cycle, auth) -> None:
    current = cycle
    for state in (
        CycleState.OBSERVING,
        CycleState.RESEARCHING,
        CycleState.DIAGNOSING,
        CycleState.SPECIFYING,
        CycleState.PLANNING,
        CycleState.PREPARING_WORKSPACE,
        CycleState.PATCHING,
        CycleState.FORMATTING,
        CycleState.LINTING,
        CycleState.TYPE_CHECKING,
        CycleState.TESTING,
    ):
        current = advance(current, state, auth, action=ThanosAction.EDIT)
    assert current.state is CycleState.TESTING
    assert auth.is_active is True


# --------------------------------------------------------------------------
# Merge gate integration
# --------------------------------------------------------------------------


def _at_merging(cycle, auth):
    current = cycle
    for state in (
        CycleState.OBSERVING,
        CycleState.RESEARCHING,
        CycleState.DIAGNOSING,
        CycleState.SPECIFYING,
        CycleState.PLANNING,
        CycleState.PREPARING_WORKSPACE,
        CycleState.PATCHING,
        CycleState.FORMATTING,
        CycleState.LINTING,
        CycleState.TYPE_CHECKING,
        CycleState.TESTING,
        CycleState.SECURITY_REVIEWING,
        CycleState.PACKAGING,
        CycleState.COMMITTING,
        CycleState.PUSHING,
        CycleState.OPENING_PR,
        CycleState.WAITING_FOR_CI,
        CycleState.MERGING,
    ):
        current = advance(current, state, auth)
    return current


def _preconditions(
    *,
    repository: str = "ProCityHub/GARVIS",
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


def test_green_cycle_merges_without_owner_checkpoint(cycle, auth) -> None:
    at_merge = _at_merging(cycle, auth)
    decision = evaluate_merge_gate(auth, _preconditions())
    merged = record_merge_decision(at_merge, decision, auth)
    assert merged.state is CycleState.SYNCHRONIZING
    assert merged.blocker is None


def test_red_ci_blocks_the_merge(cycle, auth) -> None:
    at_merge = _at_merging(cycle, auth)
    decision = evaluate_merge_gate(auth, _preconditions(required_checks_passed=False))
    blocked = record_merge_decision(at_merge, decision, auth)
    assert blocked.state is CycleState.BLOCKED
    assert "did not pass" in (blocked.blocker or "")


def test_governance_change_blocks_self_merge(cycle, auth) -> None:
    at_merge = _at_merging(cycle, auth)
    decision = evaluate_merge_gate(
        auth, _preconditions(changed_paths=("src/garvis/thanos_mode.py",))
    )
    blocked = record_merge_decision(at_merge, decision, auth)
    assert blocked.state is CycleState.BLOCKED
    assert decision.requires_owner_review is True


# --------------------------------------------------------------------------
# Durability and resume
# --------------------------------------------------------------------------


def test_interrupted_cycle_resumes(tmp_path, cycle, auth) -> None:
    store = CycleStore(tmp_path / "cycles.json")
    working = advance(cycle, CycleState.OBSERVING, auth)
    working = advance(working, CycleState.RESEARCHING, auth)
    store.save(working)

    resumed = CycleStore(tmp_path / "cycles.json").resume()
    assert resumed is not None
    assert resumed.cycle_id == working.cycle_id
    assert resumed.state is CycleState.RESEARCHING
    assert len(resumed.transitions) == 2
    assert resumed.starting_commit == "049fc9d"


def test_no_active_cycle_returns_none(tmp_path) -> None:
    assert CycleStore(tmp_path / "cycles.json").resume() is None


def test_terminal_cycle_is_archived(tmp_path, cycle, auth) -> None:
    store = CycleStore(tmp_path / "cycles.json")
    store.save(advance(cycle, CycleState.FAILED, auth))
    assert store.resume() is None
    assert len(store.history()) == 1
    assert store.history()[0].state is CycleState.FAILED


def test_corrupt_state_is_detected(tmp_path) -> None:
    path = tmp_path / "cycles.json"
    path.write_text("{broken", encoding="utf-8")
    store = CycleStore(path)
    assert store.is_corrupt() is True
    assert store.resume() is None


def test_atomic_write_leaves_no_temp_files(tmp_path, cycle) -> None:
    CycleStore(tmp_path / "cycles.json").save(cycle)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".cycle-")]
    assert leftovers == []


def test_evidence_survives_the_round_trip(tmp_path, cycle, auth) -> None:
    store = CycleStore(tmp_path / "cycles.json")
    working = advance(
        cycle,
        CycleState.OBSERVING,
        auth,
        evidence={"source_url": "https://docs.python.org/3.9/", "confidence": "high"},
        changed_files=("src/garvis/assistant.py",),
    )
    store.save(working)
    resumed = store.resume()
    assert resumed is not None
    assert resumed.evidence["confidence"] == "high"
    assert resumed.changed_files == ("src/garvis/assistant.py",)


# --------------------------------------------------------------------------
# Concurrency lock
# --------------------------------------------------------------------------


def test_second_cycle_cannot_acquire_the_lock(tmp_path) -> None:
    first = CycleLock(tmp_path / "cycle.lock").acquire()
    try:
        with pytest.raises(CycleLockError):
            CycleLock(tmp_path / "cycle.lock").acquire()
    finally:
        first.release()


def test_lock_is_reusable_after_release(tmp_path) -> None:
    path = tmp_path / "cycle.lock"
    CycleLock(path).acquire().release()
    second = CycleLock(path).acquire()
    second.release()
    assert not path.exists()


def test_stale_lock_from_dead_process_is_reaped(tmp_path) -> None:
    path = tmp_path / "cycle.lock"
    path.write_text('{"pid": 999999, "acquired_at": "2026-07-24T00:00:00Z"}')
    lock = CycleLock(path).acquire()
    lock.release()


def test_live_lock_is_not_reaped(tmp_path) -> None:
    path = tmp_path / "cycle.lock"
    path.write_text(f'{{"pid": {os.getpid()}, "acquired_at": "2026-07-24T00:00:00Z"}}')
    with pytest.raises(CycleLockError):
        CycleLock(path).acquire()


def test_lock_context_manager_releases(tmp_path) -> None:
    path = tmp_path / "cycle.lock"
    with CycleLock(path):
        assert path.exists()
    assert not path.exists()
