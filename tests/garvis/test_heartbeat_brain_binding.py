import json

import pytest

from garvis.brain_binding import HeartbeatBrainBinding
from garvis.heartbeat_service import AutomaticHeartbeatService
from garvis.self_authority import GarvisSelfAuthority, InternalAction, require_self_authority


def test_live_brain_predicts_from_verified_cycles_after_restart(tmp_path):
    """Verified learning survives restart and is frozen into the next prediction."""
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
    """An execution failure leaves learned outcomes unchanged."""
    import garvis.heartbeat_service as module

    service = AutomaticHeartbeatService(tmp_path)
    original = require_self_authority

    def deny_consolidation(authority, action):
        """Simulate denied consolidation while preserving other authority checks."""
        if action is InternalAction.CONSOLIDATE:
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
    """A contradictory return phase blocks learning and reports unhealthy status."""
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
    """An absent repository supplies contradiction rather than supporting evidence."""
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
    """Repeated observations remain one source and retain bounded history."""
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
    """Reject checkpoints claiming more verified outcomes than assessed cycles."""
    with pytest.raises(ValueError, match="counters"):
        HeartbeatBrainBinding({"version": 1, "cycle_id": 1, "verified_cycles": 2})


def test_existing_heartbeat_state_without_brain_migrates(tmp_path):
    """Legacy sequence state acquires a fresh brain checkpoint on its next cycle."""
    (tmp_path / "heartbeat_state.json").write_text('{"sequence": 7}')
    service = AutomaticHeartbeatService(tmp_path)
    try:
        service.run_once()
        assert service.sequence == 8
        assert service.health()["brain_checkpoint"]["cycle_id"] == 1
    finally:
        service.close()

@pytest.mark.parametrize("interruption", ["after_result_commit", "before_checkpoint_replace"])
@pytest.mark.parametrize("baseline", [0, 1])
def test_committed_cycle_is_recovered_once_after_interruption(
    tmp_path, monkeypatch, interruption, baseline
):
    """A durable result survives interruption without losing or duplicating learning."""
    from pathlib import Path

    service = AutomaticHeartbeatService(tmp_path)
    for _ in range(baseline):
        service.run_once()
    try:
        with monkeypatch.context() as patch:
            if interruption == "after_result_commit":
                append_result = service.predictions.append_result

                def interrupt_after_commit(*args, **kwargs):
                    """Stop after SQLite has committed but before learning starts."""
                    append_result(*args, **kwargs)
                    raise KeyboardInterrupt("simulated process interruption")

                patch.setattr(service.predictions, "append_result", interrupt_after_commit)
            else:
                replace = Path.replace

                def interrupt_replace(path, target):
                    """Leave a complete temporary file but do not replace the checkpoint."""
                    if path == service.state_path.with_suffix(".tmp"):
                        raise KeyboardInterrupt("simulated process interruption")
                    return replace(path, target)

                patch.setattr(Path, "replace", interrupt_replace)
            with pytest.raises(KeyboardInterrupt):
                service.run_once()
        result_count = service.predictions.db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        assert result_count == baseline + 1
        if baseline:
            checkpoint = json.loads(service.state_path.read_text())["brain_checkpoint"]
            assert checkpoint["verified_cycles"] == baseline
        else:
            assert not service.state_path.exists()
    finally:
        service.close()

    for _ in range(2):
        restarted = AutomaticHeartbeatService(tmp_path)
        try:
            assert restarted.sequence == baseline + 1
            assert restarted.brain.checkpoint() == {
                "version": 1, "cycle_id": baseline + 1, "verified_cycles": baseline + 1
            }
            assert restarted.health()["brain_checkpoint"]["verified_cycles"] == baseline + 1
        finally:
            restarted.close()
    restarted = AutomaticHeartbeatService(tmp_path)
    try:
        next_cycle = restarted.run_once()
        assert next_cycle.prediction["next_sequence"] == baseline + 2
        assert next_cycle.prediction["brain"]["prediction"] == HeartbeatBrainBinding.OUTCOME
        assert restarted.brain.checkpoint()["verified_cycles"] == baseline + 2
    finally:
        restarted.close()

@pytest.mark.parametrize("outcome", ["failed", "contradictory", "uncommitted"])
def test_recovery_does_not_learn_unverified_results(tmp_path, monkeypatch, outcome):
    """Recovery cannot turn failure, contradiction, or a prediction into learning."""
    import garvis.heartbeat_service as module

    service = AutomaticHeartbeatService(tmp_path)
    try:
        with monkeypatch.context() as patch:
            append_result = service.predictions.append_result

            def interrupt_result(*args, **kwargs):
                """Interrupt on the chosen side of the result commit boundary."""
                if outcome != "uncommitted":
                    append_result(*args, **kwargs)
                raise KeyboardInterrupt("interrupted")

            def deny_execution(authority, action):
                """Deny consolidation while retaining the other authority checks."""
                if action is InternalAction.CONSOLIDATE:
                    raise PermissionError("denied")
                return require_self_authority(authority, action)

            patch.setattr(service.predictions, "append_result", interrupt_result)
            if outcome == "failed":
                patch.setattr(module, "require_self_authority", deny_execution)
            if outcome == "contradictory":
                patch.setattr(module, "oab_wrap_phase_index", lambda _: 1)
            with pytest.raises(KeyboardInterrupt):
                service.run_once()
    finally:
        service.close()
    restarted = AutomaticHeartbeatService(tmp_path)
    try:
        assert restarted.brain.checkpoint()["verified_cycles"] == 0
        assert restarted.sequence == (1 if outcome == "contradictory" else 0)
        assert restarted.health()["heartbeat_healthy"] is False
    finally:
        restarted.close()


def test_recovery_requires_learning_authority(tmp_path, monkeypatch):
    """A committed cycle does not grant permission to train on restart."""
    service = AutomaticHeartbeatService(tmp_path)
    try:
        def interrupt_write(payload):
            """Leave the committed result without a saved checkpoint."""
            raise KeyboardInterrupt("interrupted")

        monkeypatch.setattr(service, "_write_state", interrupt_write)
        with pytest.raises(KeyboardInterrupt):
            service.run_once()
    finally:
        service.close()
    authority = GarvisSelfAuthority(allowed_actions=(InternalAction.HEARTBEAT.value,))
    with pytest.raises(PermissionError):
        AutomaticHeartbeatService(tmp_path, self_authority=authority)
    restarted = AutomaticHeartbeatService(tmp_path)
    try:
        assert restarted.brain.checkpoint()["verified_cycles"] == 1
    finally:
        restarted.close()


def test_legacy_checkpoint_cycle_identity_prevents_duplicate_learning(tmp_path):
    """Checkpoints predating the result cursor resume after their saved cycle UUID."""
    service = AutomaticHeartbeatService(tmp_path)
    try:
        service.run_once()
        saved = json.loads(service.state_path.read_text())
        del saved["brain_result_rowid"]
        service.state_path.write_text(json.dumps(saved))
    finally:
        service.close()
    restarted = AutomaticHeartbeatService(tmp_path)
    try:
        restarted.run_once()
        assert restarted.brain.checkpoint()["verified_cycles"] == 2
    finally:
        restarted.close()
