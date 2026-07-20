import pytest

from garvis.lattice_cognition import (
    RecurrentRecallResult,
    evaluate_recall_equilibrium,
)


def build_recall(
    *,
    similarity: float = 1.0,
    recalled: bool = True,
    converged: bool = True,
    final_delta: float = 0.0,
) -> RecurrentRecallResult:
    return RecurrentRecallResult(
        cycles_run=4,
        converged=converged,
        final_delta=final_delta,
        final_state=(
            ("evidence:center", 1.0),
            ("evidence:observed:result", similarity),
        ),
        recalled_attractor_id=(
            "evidence-attractor:test"
            if recalled
            else None
        ),
        similarity=similarity,
        recalled=recalled,
        lattice_sha256="a" * 64,
        pulse_sha256="b" * 64,
    )


def test_complete_recall_reaches_corner_seven() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(),
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=True,
    )

    equilibrium = assessment.equilibrium

    assert assessment.recall_quality == pytest.approx(1.0)
    assert assessment.convergence_quality == pytest.approx(1.0)
    assert equilibrium.raw_union == pytest.approx(1.6)
    assert equilibrium.normalized_center == pytest.approx(1.0)
    assert equilibrium.corner_bits == (1, 1, 1)
    assert equilibrium.corner_index == 7
    assert equilibrium.equilibrium_reached is True
    assert equilibrium.proposal_eligible is True
    assert equilibrium.human_approval_required is True
    assert assessment.external_action_allowed is False


def test_failed_recall_blocks_equilibrium() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(
            similarity=1.0,
            recalled=False,
        ),
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    assert assessment.recall_quality == pytest.approx(0.0)
    assert assessment.constraints_passed is False
    assert assessment.equilibrium.equilibrium_reached is False
    assert assessment.equilibrium.proposal_eligible is False


def test_nonconverged_recall_blocks_equilibrium() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(
            converged=False,
            final_delta=0.2,
        ),
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    assert assessment.convergence_quality == pytest.approx(0.8)
    assert assessment.constraints_passed is False
    assert assessment.equilibrium.equilibrium_reached is False


def test_evidence_remains_a_limiting_dimension() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(),
        evidence_sufficiency=0.4,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    assert assessment.evidence_quality == pytest.approx(0.4)
    assert assessment.equilibrium.corner_bits[0] == 0
    assert assessment.equilibrium.proposal_eligible is False


def test_similarity_limits_action_readiness() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(similarity=0.8),
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    assert assessment.action_quality == pytest.approx(0.8)
    assert assessment.equilibrium.proposal_eligible is False


def test_assessment_hash_is_deterministic() -> None:
    recall = build_recall()

    first = evaluate_recall_equilibrium(
        recall=recall,
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    second = evaluate_recall_equilibrium(
        recall=recall,
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=False,
    )

    assert first.recall_sha256 == second.recall_sha256
    assert first == second


def test_invalid_unit_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_recall_equilibrium(
            recall=build_recall(),
            evidence_sufficiency=1.1,
            action_readiness=1.0,
            context_stability=1.0,
            constraints_passed=True,
            external_action=False,
        )


def test_assessment_contains_no_execution_interface() -> None:
    assessment = evaluate_recall_equilibrium(
        recall=build_recall(),
        evidence_sufficiency=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        constraints_passed=True,
        external_action=True,
    )

    assert assessment.external_action_allowed is False
    assert not hasattr(assessment, "execute")
    assert not hasattr(assessment, "send")
    assert not hasattr(assessment, "connect")
