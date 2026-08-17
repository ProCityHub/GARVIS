import pytest

from garvis.evidence_envelope import build_evidence_envelope
from garvis.memory_lifecycle import MemoryControlSignal
from garvis.lattice_cognition import (
    CognitiveCycleStage,
    PulsePhase,
    run_lattice_cognitive_cycle,
)


def build_strong_envelope():
    return build_evidence_envelope(
        immutable_source_evidence={
            "ultimatum-result": {
                "status": "negative",
                "auc_phi": 0.871258,
                "auc_flat": 0.861944,
            }
        },
        deterministic_agi_measurements={
            "difference": 0.009314,
        },
    )


def build_low_quality_envelope():
    return build_evidence_envelope(
        immutable_source_evidence={},
        symbolic_interpretation={
            "cube-mapping": "engineering metaphor",
        },
        unsupported_speculation={
            "cause": "unverified mechanism",
        },
        unknowns={
            "mechanism": "not established",
        },
    )


def test_complete_cycle_reaches_human_review_boundary() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=1,
        external_action=True,
    )

    equilibrium = result.recall_equilibrium.equilibrium

    assert result.assessment.evidence_sufficiency == (
        pytest.approx(1.0)
    )
    assert result.pulse.phase is PulsePhase.ACTIVATE
    assert result.pulse.raw_union == pytest.approx(1.6)
    assert result.pulse.normalized_center == pytest.approx(1.0)
    assert result.recall.recalled is True
    assert result.recall.converged is True
    assert equilibrium.raw_union > 1.59
    assert equilibrium.normalized_center > 0.99
    assert equilibrium.corner_index == 7
    assert equilibrium.equilibrium_reached is True
    assert result.proposal_eligible is True
    assert result.human_approval_required is True
    assert result.decision == "HUMAN_REVIEW_REQUIRED"
    assert result.external_action_allowed is False


def test_low_quality_evidence_blocks_proposal() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_low_quality_envelope(),
        cycle=1,
    )

    assert result.assessment.evidence_sufficiency < 0.25
    assert result.proposal_eligible is False
    assert result.decision == "BLOCKED"
    assert result.external_action_allowed is False


def test_partial_cue_propagates_through_memory() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=2,
    )

    selected = (result.cue_signal_ids[0],)

    partial = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=2,
        cue_signal_ids=selected,
    )

    state = partial.recall.state_dict()

    assert partial.cue_signal_ids == selected
    assert state[partial.consolidated_memory.center_node] > 0.0
    assert sum(value > 0.0 for value in state.values()) >= 2
    assert partial.completed_stages[-1] is (
        CognitiveCycleStage.EVALUATE_PROPOSAL
    )


def test_unknown_cue_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_lattice_cognitive_cycle(
            envelope=build_strong_envelope(),
            cycle=1,
            cue_signal_ids=("unknown-signal",),
        )


def test_duplicate_cues_are_rejected() -> None:
    baseline = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=1,
    )

    signal_id = baseline.cue_signal_ids[0]

    with pytest.raises(ValueError):
        run_lattice_cognitive_cycle(
            envelope=build_strong_envelope(),
            cycle=1,
            cue_signal_ids=(signal_id, signal_id),
        )


def test_empty_envelope_cannot_form_memory() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={},
    )

    with pytest.raises(ValueError):
        run_lattice_cognitive_cycle(
            envelope=envelope,
            cycle=1,
        )


def test_cycle_is_deterministic() -> None:
    first = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=7,
        external_action=True,
    )

    second = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=7,
        external_action=True,
    )

    assert first.envelope_sha256 == second.envelope_sha256
    assert first.cycle_sha256 == second.cycle_sha256
    assert first.recall == second.recall
    assert first.recall_equilibrium == second.recall_equilibrium


def test_cycle_number_changes_cycle_hash() -> None:
    first = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=1,
    )

    second = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=2,
    )

    assert first.pulse.compute_sha256() != (
        second.pulse.compute_sha256()
    )
    assert first.cycle_sha256 != second.cycle_sha256


def _memory_control(
    *,
    foreground_required: bool = False,
    contradiction_observed: bool = False,
    execution_authority: bool = False,
    silent_consolidation_allowed: bool = False,
    reason: str = "procedural_candidate_available",
) -> MemoryControlSignal:
    return MemoryControlSignal(
        cue="memory control test cue",
        procedural_candidates=(),
        prospective_triggers=(),
        contradiction_observed=contradiction_observed,
        foreground_required=foreground_required,
        execution_authority=execution_authority,
        silent_consolidation_allowed=silent_consolidation_allowed,
        reason=reason,
    )


def test_memory_control_does_not_rewrite_evidence_or_consolidation() -> None:
    baseline = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=11,
    )

    controlled = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=11,
        memory_control=_memory_control(),
    )

    assert controlled.envelope_sha256 == baseline.envelope_sha256
    assert controlled.assessment == baseline.assessment
    assert (
        controlled.consolidated_memory.memory.compute_sha256()
        == baseline.consolidated_memory.memory.compute_sha256()
    )
    assert controlled.recall == baseline.recall
    assert (
        controlled.evidence_proposal_eligible
        == baseline.evidence_proposal_eligible
    )
    assert controlled.decision == baseline.decision
    assert controlled.external_action_allowed is False


def test_memory_foreground_signal_interrupts_proposal_only() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=12,
        memory_control=_memory_control(
            foreground_required=True,
            reason="prospective_cue",
        ),
    )

    assert result.evidence_proposal_eligible is True
    assert result.proposal_eligible is False
    assert result.decision == "MEMORY_FOREGROUND_REVIEW_REQUIRED"
    assert result.external_action_allowed is False


def test_contradiction_memory_signal_requires_foreground_review() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=13,
        memory_control=_memory_control(
            foreground_required=True,
            contradiction_observed=True,
            reason="contradiction",
        ),
    )

    assert result.memory_control is not None
    assert result.memory_control.contradiction_observed is True
    assert result.proposal_eligible is False
    assert result.decision == "MEMORY_FOREGROUND_REVIEW_REQUIRED"
    assert result.external_action_allowed is False


def test_memory_control_cannot_grant_execution_authority() -> None:
    with pytest.raises(
        ValueError,
        match="memory control cannot grant execution authority",
    ):
        run_lattice_cognitive_cycle(
            envelope=build_strong_envelope(),
            cycle=14,
            memory_control=_memory_control(
                execution_authority=True,
            ),
        )


def test_memory_control_cannot_enable_silent_consolidation() -> None:
    with pytest.raises(
        ValueError,
        match="memory control cannot authorize silent consolidation",
    ):
        run_lattice_cognitive_cycle(
            envelope=build_strong_envelope(),
            cycle=15,
            memory_control=_memory_control(
                silent_consolidation_allowed=True,
            ),
        )


def test_memory_control_stage_precedes_proposal_stage() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=16,
        memory_control=_memory_control(),
    )

    memory_index = result.completed_stages.index(
        CognitiveCycleStage.EVALUATE_MEMORY_CONTROL
    )
    proposal_index = result.completed_stages.index(
        CognitiveCycleStage.EVALUATE_PROPOSAL
    )

    assert memory_index < proposal_index


def test_memory_control_is_bound_into_cycle_hash_not_evidence_hash() -> None:
    baseline = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=17,
    )

    interrupted = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=17,
        memory_control=_memory_control(
            foreground_required=True,
            reason="prospective_cue",
        ),
    )

    assert interrupted.envelope_sha256 == baseline.envelope_sha256
    assert interrupted.assessment == baseline.assessment
    assert interrupted.cycle_sha256 != baseline.cycle_sha256


def test_cycle_has_no_execution_interface() -> None:
    result = run_lattice_cognitive_cycle(
        envelope=build_strong_envelope(),
        cycle=1,
        external_action=True,
    )

    assert result.external_action_allowed is False
    assert not hasattr(result, "execute")
    assert not hasattr(result, "send")
    assert not hasattr(result, "connect")
