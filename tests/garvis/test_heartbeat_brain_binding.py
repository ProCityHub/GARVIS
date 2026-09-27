import json

import pytest

from garvis.brain_binding import HeartbeatBrainBinding
from garvis.heartbeat_service import AutomaticHeartbeatService


def test_live_brain_predicts_from_verified_cycles_after_restart(tmp_path):
    service = AutomaticHeartbeatService(tmp_path)
    try:
        first = service.run_once()
        assert first.status.value == "completed"
        assert first.prediction["brain"]["prediction"] is None
        assert first.raw_post["brain"]["cycle_id"] == 1
        assert service.health()["brain_checkpoint"]["verified_cycles"] == 1
        assert service.health()["brain"]["external_action_allowed"] is False
    finally:
        service.close()

    restarted = AutomaticHeartbeatService(tmp_path)
    try:
        second = restarted.run_once()
        assert second.prediction["brain"]["prediction"] == HeartbeatBrainBinding.OUTCOME
        assert second.raw_post["brain"]["cycle_id"] == 2
        assert restarted.health()["brain_checkpoint"]["verified_cycles"] == 2
        rows = restarted.predictions.db.execute(
            "SELECT payload_json FROM frozen_predictions ORDER BY rowid"
        ).fetchall()
        assert json.loads(rows[1][0])["brain"] == second.prediction["brain"]
    finally:
        restarted.close()


def test_failed_execution_does_not_train_brain(tmp_path, monkeypatch):
    import garvis.heartbeat_service as module

    service = AutomaticHeartbeatService(tmp_path)
    original = module.require_self_authority

    def deny_consolidation(authority, action):
        if action is module.InternalAction.CONSOLIDATE:
            raise PermissionError("test execution failure")
        return original(authority, action)

    monkeypatch.setattr(module, "require_self_authority", deny_consolidation)
    try:
        state = service.run_once()
        assert state.status.value == "failed"
        assert service.brain.checkpoint()["verified_cycles"] == 0
        assert service.health()["heartbeat_healthy"] is False
    finally:
        service.close()


def test_contradictory_verification_does_not_train_brain(tmp_path, monkeypatch):
    import garvis.heartbeat_service as module

    service = AutomaticHeartbeatService(tmp_path)
    monkeypatch.setattr(module, "oab_wrap_phase_index", lambda _: 1)
    try:
        state = service.run_once()
        assert state.contradictions
        assert service.brain.checkpoint()["verified_cycles"] == 0
        assert service.health()["heartbeat_healthy"] is False
    finally:
        service.close()


def test_missing_repository_is_not_supporting_evidence(tmp_path):
    service = AutomaticHeartbeatService(tmp_path / "state", repository_root=tmp_path / "absent")
    try:
        state = service.run_once()
        brain = state.raw_post["brain"]
        assert brain["truth_state"] == "CONTRADICTED"
        assert brain["support"] == 0
        assert brain["action_decision"] == "OBSERVE_MORE"
        assert brain["external_action_allowed"] is False
        assert not (tmp_path / "absent").exists()
    finally:
        service.close()


def test_repetition_is_not_independent_evidence_and_memory_is_bounded():
    binding = HeartbeatBrainBinding()
    for _ in range(150):
        result = binding.assess({"repository_available": True})
        binding.learn_verified_cycle()
    assert result["truth_state"] == "SUPPORTED"
    assert len(binding.engine.working) == 8
    assert len(binding.engine.episodes) <= 89
    assert len(binding.engine.events) <= 89
    assert binding.engine.semantic == {}
    assert binding.checkpoint()["verified_cycles"] == 150


def test_invalid_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="counters"):
        HeartbeatBrainBinding({"version": 1, "cycle_id": 1, "verified_cycles": 2})


def test_existing_heartbeat_state_without_brain_migrates(tmp_path):
    (tmp_path / "heartbeat_state.json").write_text('{"sequence": 7}')
    service = AutomaticHeartbeatService(tmp_path)
    try:
        service.run_once()
        assert service.sequence == 8
        assert service.health()["brain_checkpoint"]["cycle_id"] == 1
    finally:
        service.close()
