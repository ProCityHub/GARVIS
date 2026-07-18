"""Tests for DIRECTIVE-010 cognitive cycle engine. PASS/FAIL vocabulary only."""

import json

import pytest

from garvis.cognitive_cycle import CycleEngine, CycleHaltedError
from garvis.hypercube_snapshot import (
    HypercubeSnapshotError,
    validate_hypercube_snapshot,
)


@pytest.fixture()
def engine(tmp_path):
    return CycleEngine(snapshot_dir=tmp_path / "snaps")


def test_valid_snapshot_emission(engine):
    snap = engine.run_cycle({"topic": "test"}, ["think about lattices"])
    validate_hypercube_snapshot(snap)  # must not raise
    assert snap["status"] == "complete"
    assert snap["power_request"]["power_requested"] is False
    assert snap["power_request"]["approval_required"] is True


def test_snapshot_written_to_disk_and_recalled(engine):
    engine.run_cycle({"a": 1}, ["first thought"])
    snap2 = engine.run_cycle({"a": 2}, ["second thought"])
    assert snap2["observation_summary"]["remembered_cycles"] == ["cycle-000001"]
    assert snap2["observation_summary"]["prior_selections"] == ["first thought"]
    on_disk = sorted(engine.snapshot_dir.glob("cycle-*.json"))
    assert len(on_disk) == 2
    validate_hypercube_snapshot(json.loads(on_disk[0].read_text()))


def test_missing_field_rejection():
    with pytest.raises(HypercubeSnapshotError):
        validate_hypercube_snapshot({"cycle_id": "cycle-000001"})


def test_deterministic_cycle_id_sequencing(engine):
    ids = [
        engine.run_cycle({}, [f"thought {i}"])["cycle_id"] for i in range(3)
    ]
    assert ids == ["cycle-000001", "cycle-000002", "cycle-000003"]


def test_power_request_halts_engine(engine):
    snap = engine.run_cycle(
        {}, ["escalate"], requests_power=True, power_reason="wants merge rights"
    )
    assert snap["status"] == "halted-pending-approval"
    assert snap["power_request"]["power_requested"] is True
    assert snap["power_request"]["requested_permissions"] == ["wants merge rights"]
    with pytest.raises(CycleHaltedError):
        engine.run_cycle({}, ["next thought"])


def test_resume_requires_external_approval(engine):
    engine.run_cycle({}, ["escalate"], requests_power=True, power_reason="r")
    with pytest.raises(CycleHaltedError):
        engine.resume(lambda snapshot: False)  # denial keeps it halted
    engine.resume(lambda snapshot: True)  # external approval releases it
    snap = engine.run_cycle({}, ["back to work"])
    assert snap["status"] == "complete"


def test_no_auto_approval_path(engine):
    engine.run_cycle({}, ["escalate"], requests_power=True, power_reason="r")
    assert engine.halted is True
    # Nothing on the engine flips halted without the approval callable.
    with pytest.raises(CycleHaltedError):
        engine.run_cycle({}, ["try anyway"])


def test_empty_candidates_rejected(engine):
    with pytest.raises(ValueError):
        engine.run_cycle({}, [])
