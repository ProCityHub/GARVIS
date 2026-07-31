import pytest

from garvis.evidence_envelope import build_evidence_envelope
from garvis.lattice_cognition import (
    PulseBus,
    consolidate_evidence_assessment,
)
from garvis.psychology import (
    EvidenceSignalKind,
    adapt_evidence_envelope,
)


def build_assessment():
    envelope = build_evidence_envelope(
        immutable_source_evidence={
            "ultimatum-result": {
                "result": "negative",
                "auc_phi": 0.871258,
                "auc_flat": 0.861944,
            }
        },
        deterministic_agi_measurements={
            "difference": 0.009314,
        },
        engineering_inference={
            "interpretation": (
                "The registered margin was not reached."
            ),
        },
        symbolic_interpretation={
            "cube-mapping": "engineering metaphor only",
        },
        unsupported_speculation={
            "unverified-cause": "quantum explanation",
        },
        unknowns={
            "mechanism": "not established",
        },
    )

    return adapt_evidence_envelope(envelope)


def test_consolidation_filters_weak_signals() -> None:
    assessment = build_assessment()

    consolidated = consolidate_evidence_assessment(
        assessment,
        minimum_routing_weight=0.25,
    )

    retained = set(consolidated.retained_signal_ids)

    for signal in assessment.signals:
        if signal.kind in {
            EvidenceSignalKind.SPECULATIVE,
            EvidenceSignalKind.UNKNOWN,
        }:
            assert signal.signal_id not in retained
        else:
            assert signal.signal_id in retained


def test_consolidation_preserves_provenance() -> None:
    assessment = build_assessment()
    consolidated = consolidate_evidence_assessment(
        assessment
    )

    attractor = consolidated.memory.attractors[0]

    expected = {
        signal.provenance_hash
        for signal in assessment.signals
        if signal.signal_id
        in consolidated.retained_signal_ids
    }

    assert set(attractor.provenance_refs) == expected


def test_consolidation_is_deterministic() -> None:
    assessment = build_assessment()

    first = consolidate_evidence_assessment(
        assessment
    )

    second = consolidate_evidence_assessment(
        assessment
    )

    assert first.assessment_sha256 == (
        second.assessment_sha256
    )

    assert first.memory.compute_sha256() == (
        second.memory.compute_sha256()
    )


def test_single_signal_cue_propagates_through_center() -> None:
    assessment = build_assessment()
    consolidated = consolidate_evidence_assessment(
        assessment
    )

    signal_id = consolidated.retained_signal_ids[0]

    result = consolidated.recall(
        signal_ids=(signal_id,),
        pulse=PulseBus().emit(cycle=1),
    )

    state = result.state_dict()

    assert state[consolidated.center_node] > 0.0

    activated_signal_nodes = [
        value
        for signal_node, value in state.items()
        if signal_node != consolidated.center_node
        and value > 0.0
    ]

    assert len(activated_signal_nodes) >= 2
    assert result.external_action_allowed is False


def test_unconsolidated_signal_is_rejected() -> None:
    assessment = build_assessment()
    consolidated = consolidate_evidence_assessment(
        assessment
    )

    excluded_id = consolidated.excluded_signal_ids[0]

    with pytest.raises(ValueError):
        consolidated.recall(
            signal_ids=(excluded_id,),
            pulse=PulseBus().emit(cycle=1),
        )


def test_empty_signal_cue_is_rejected() -> None:
    assessment = build_assessment()
    consolidated = consolidate_evidence_assessment(
        assessment
    )

    with pytest.raises(ValueError):
        consolidated.recall(
            signal_ids=(),
            pulse=PulseBus().emit(cycle=1),
        )


def test_empty_assessment_cannot_form_memory() -> None:
    assessment = adapt_evidence_envelope(
        build_evidence_envelope(
            immutable_source_evidence={},
        )
    )

    with pytest.raises(ValueError):
        consolidate_evidence_assessment(
            assessment
        )


def test_consolidated_memory_has_no_execution_interface() -> None:
    consolidated = consolidate_evidence_assessment(
        build_assessment()
    )

    assert consolidated.external_action_allowed is False
    assert not hasattr(consolidated, "execute")
    assert not hasattr(consolidated, "send")
    assert not hasattr(consolidated, "connect")
