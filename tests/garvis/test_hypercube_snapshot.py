import json
from pathlib import Path

import pytest

from garvis.hypercube_snapshot import (
    HypercubeSnapshotError,
    load_hypercube_snapshot,
    validate_hypercube_snapshot,
)


def valid_snapshot() -> dict:
    return {
        "cycle_id": "cycle-001",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "stage 2 cognitive draft",
        "operator_context": {},
        "input_state": {},
        "observation_summary": {},
        "candidate_thoughts": [],
        "comparison": {},
        "selection": {},
        "uncertainty": {},
        "evolution_contract": {},
        "next_smallest_step": {},
        "output_boundary": {},
    }


def test_valid_snapshot_is_accepted() -> None:
    snapshot = valid_snapshot()

    validated = validate_hypercube_snapshot(snapshot)

    assert validated == snapshot
    assert validated is not snapshot


def test_missing_required_field_is_rejected() -> None:
    snapshot = valid_snapshot()
    del snapshot["output_boundary"]

    with pytest.raises(
        HypercubeSnapshotError,
        match="missing required fields: output_boundary",
    ):
        validate_hypercube_snapshot(snapshot)


def test_snapshot_file_is_loaded_read_only(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "latest_cognitive_cycle.json"
    original_text = json.dumps(valid_snapshot(), sort_keys=True)
    snapshot_path.write_text(original_text, encoding="utf-8")

    loaded = load_hypercube_snapshot(snapshot_path)

    assert loaded["cycle_id"] == "cycle-001"
    assert snapshot_path.read_text(encoding="utf-8") == original_text


def test_missing_snapshot_file_has_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "latest_cognitive_cycle.json"

    with pytest.raises(HypercubeSnapshotError, match="file was not found"):
        load_hypercube_snapshot(missing_path)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "latest_cognitive_cycle.json"
    snapshot_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(HypercubeSnapshotError, match="not valid JSON"):
        load_hypercube_snapshot(snapshot_path)
