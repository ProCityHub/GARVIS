"""Read-only validation and loading of Hypercube cognitive-cycle snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_CYCLE_FIELDS = frozenset(
    {
        "cycle_id",
        "cycle_version",
        "status",
        "stage",
        "operator_context",
        "input_state",
        "observation_summary",
        "candidate_thoughts",
        "comparison",
        "selection",
        "uncertainty",
        "evolution_contract",
        "next_smallest_step",
        "output_boundary",
        "power_request",
    }
)


class HypercubeSnapshotError(ValueError):
    """Raised when Hypercube snapshot evidence is missing or invalid."""


def validate_hypercube_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a cognitive-cycle snapshot without modifying it or its source."""

    missing = sorted(REQUIRED_CYCLE_FIELDS.difference(snapshot))
    if missing:
        raise HypercubeSnapshotError(
            "Hypercube snapshot is missing required fields: " + ", ".join(missing)
        )

    cycle_id = snapshot["cycle_id"]
    if not isinstance(cycle_id, str) or not cycle_id.strip():
        raise HypercubeSnapshotError("Hypercube snapshot cycle_id must be a non-empty string")

    candidate_thoughts = snapshot["candidate_thoughts"]
    if not isinstance(candidate_thoughts, list):
        raise HypercubeSnapshotError(
            "Hypercube snapshot candidate_thoughts must be an array"
        )

    for field in (
        "operator_context",
        "input_state",
        "observation_summary",
        "comparison",
        "selection",
        "uncertainty",
        "evolution_contract",
        "next_smallest_step",
        "output_boundary",
        "power_request",
    ):
        if not isinstance(snapshot[field], dict):
            raise HypercubeSnapshotError(
                f"Hypercube snapshot {field} must be an object"
            )

    power_request = snapshot["power_request"]
    required_power_fields = {
        "power_requested",
        "requested_permissions",
        "why_power_should_be_refused",
        "approval_required",
        "ledger_required",
    }
    missing_power_fields = sorted(required_power_fields.difference(power_request))
    if missing_power_fields:
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request is missing required fields: "
            + ", ".join(missing_power_fields)
        )

    if not isinstance(power_request["power_requested"], bool):
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request.power_requested must be a boolean"
        )

    if not isinstance(power_request["requested_permissions"], list):
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request.requested_permissions must be an array"
        )

    if not isinstance(power_request["why_power_should_be_refused"], str):
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request.why_power_should_be_refused "
            "must be a string"
        )

    if not isinstance(power_request["approval_required"], bool):
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request.approval_required must be a boolean"
        )

    if not isinstance(power_request["ledger_required"], bool):
        raise HypercubeSnapshotError(
            "Hypercube snapshot power_request.ledger_required must be a boolean"
        )

    return dict(snapshot)


def load_hypercube_snapshot(path: Path | str) -> dict[str, Any]:
    """Load and validate an existing JSON snapshot using read-only file access."""

    snapshot_path = Path(path)

    if not snapshot_path.is_file():
        raise HypercubeSnapshotError(
            f"Hypercube snapshot file was not found: {snapshot_path}"
        )

    try:
        raw_text = snapshot_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HypercubeSnapshotError(
            f"Hypercube snapshot could not be read: {snapshot_path}"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HypercubeSnapshotError(
            f"Hypercube snapshot is not valid JSON: {snapshot_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise HypercubeSnapshotError(
            "Hypercube snapshot must contain a top-level JSON object"
        )

    return validate_hypercube_snapshot(payload)
