import pytest

from garvis.psychology import evaluate_equilibrium, unified_center


def test_one_plus_point_six_normalization() -> None:
    assert 1.0 + 0.6 == pytest.approx(1.6)
    assert unified_center(1.0, 0.6) == pytest.approx(1.0)


def test_complete_equilibrium_reaches_corner_seven() -> None:
    result = evaluate_equilibrium(
        activation=1.0,
        wall_coherence=0.6,
        evidence=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        tensions={
            "contradiction": 0.0,
            "uncertainty": 0.0,
            "memory_conflict": 0.0,
            "self_model_error": 0.0,
            "relational_risk": 0.0,
        },
        constraints_passed=True,
        external_action=True,
    )

    assert result.raw_union == pytest.approx(1.6)
    assert result.normalized_center == pytest.approx(1.0)
    assert result.integrated_equilibrium == pytest.approx(1.0)
    assert result.corner_bits == (1, 1, 1)
    assert result.corner_index == 7
    assert result.equilibrium_reached is True
    assert result.proposal_eligible is True
    assert result.human_approval_required is True


def test_contradiction_blocks_equilibrium() -> None:
    result = evaluate_equilibrium(
        activation=1.0,
        wall_coherence=0.6,
        evidence=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        tensions={"contradiction": 1.0, "uncertainty": 0.8},
        constraints_passed=True,
        external_action=False,
    )

    assert result.limiting_dimension == "contradiction"
    assert result.equilibrium_reached is False
    assert result.proposal_eligible is False


def test_failed_constraint_blocks_proposal() -> None:
    result = evaluate_equilibrium(
        activation=1.0,
        wall_coherence=0.6,
        evidence=1.0,
        action_readiness=1.0,
        context_stability=1.0,
        tensions={},
        constraints_passed=False,
        external_action=False,
    )

    assert result.integrated_equilibrium == pytest.approx(1.0)
    assert result.equilibrium_reached is False
