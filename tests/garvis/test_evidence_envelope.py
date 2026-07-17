import pytest
from garvis.evidence_envelope import build_evidence_envelope, EvidenceEnvelope, validate_snapshot
from typing import Any


def valid_snapshot() -> dict[str, Any]:
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
        "power_request": {
            "power_requested": False,
            "requested_permissions": [],
            "why_power_should_be_refused": "",
            "approval_required": False,
            "ledger_required": False,
        },
    }


def test_build_basic_envelope():
    immutable_data = {"source": "immutable source data"}
    hypercube_data = valid_snapshot()

    envelope = build_evidence_envelope(
        immutable_source_evidence=immutable_data,
        hypercube_cycle_data=hypercube_data,
    )

    assert isinstance(envelope, EvidenceEnvelope)
    assert envelope.immutable_source_evidence == immutable_data
    assert envelope.hypercube_cycle_data == hypercube_data


def test_invalid_hypercube_raises():
    with pytest.raises(ValueError):
        build_evidence_envelope(
            immutable_source_evidence={},
            hypercube_cycle_data={"cycle_id": "incomplete"},
        )


def test_source_provenance_hashing():
    immutable_data = {"field": "value"}

    envelope = build_evidence_envelope(immutable_source_evidence=immutable_data)

    hashed = envelope.compute_sha256_hash(immutable_data)

    new_envelope = envelope.with_source_provenance("test_source", immutable_data)

    assert "test_source" in new_envelope.source_provenance_hashes
    assert new_envelope.source_provenance_hashes["test_source"] == hashed
    assert new_envelope != envelope


def test_immutable_properties():
    envelope = build_evidence_envelope(immutable_source_evidence={})
    with pytest.raises(Exception):
        envelope.immutable_source_evidence["new"] = "forbidden"
    with pytest.raises(Exception):
        envelope.hypercube_cycle_data = {}
