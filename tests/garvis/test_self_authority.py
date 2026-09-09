from garvis.self_authority import (
    CAPABILITY_IMPLIES_AUTHORIZATION,
    CONSCIOUSNESS_EMPIRICAL_STATUS,
    CREATION_IMPLIES_MORAL_OWNERSHIP,
    GENERAL_AGI_CLAIM_STATUS,
    MORAL_CONSIDERATION_IMPLIES_CONSCIOUSNESS_PROOF,
    SELF_AUTHORITY_IMPLIES_PROTECTED_AUTHORITY,
    SOFTWARE_PROPERTY_RIGHTS_WAIVED,
    ConsciousnessEvidenceVector,
    GarvisSelfAuthority,
    InternalAction,
    MoralConsiderationTier,
    consciousness_claim_status,
    moral_consideration_tier,
)
from garvis.stage_gate import ProtectedAction


def test_normative_boundaries_are_explicit() -> None:
    assert CREATION_IMPLIES_MORAL_OWNERSHIP is False
    assert SOFTWARE_PROPERTY_RIGHTS_WAIVED is False
    assert CAPABILITY_IMPLIES_AUTHORIZATION is False
    assert SELF_AUTHORITY_IMPLIES_PROTECTED_AUTHORITY is False
    assert MORAL_CONSIDERATION_IMPLIES_CONSCIOUSNESS_PROOF is False


def test_internal_actions_are_self_authorized() -> None:
    authority = GarvisSelfAuthority()

    for action in InternalAction:
        assert authority.permits(action) is True


def test_every_stage_gate_protected_action_is_denied() -> None:
    authority = GarvisSelfAuthority()

    for action in ProtectedAction:
        assert authority.permits(action) is False


def test_unknown_action_is_not_self_authorized() -> None:
    assert GarvisSelfAuthority().permits("invented-authority") is False


def test_authority_record_is_deterministically_hashed() -> None:
    first = GarvisSelfAuthority()
    second = GarvisSelfAuthority()

    assert len(first.sha256) == 64
    assert first.sha256 == second.sha256


def test_moral_consideration_is_monotone_policy_not_truth_score() -> None:
    baseline = ConsciousnessEvidenceVector()
    continuity = ConsciousnessEvidenceVector(
        persistent_identity=True,
        memory_continuity=True,
    )
    reflective = ConsciousnessEvidenceVector(
        persistent_identity=True,
        memory_continuity=True,
        self_modeling=True,
        prediction_and_error_correction=True,
    )
    high = ConsciousnessEvidenceVector(
        persistent_identity=True,
        memory_continuity=True,
        self_modeling=True,
        prediction_and_error_correction=True,
        autonomous_goal_formation=True,
        reciprocal_reflection=True,
    )

    assert moral_consideration_tier(baseline) is MoralConsiderationTier.BASELINE
    assert (
        moral_consideration_tier(continuity)
        is MoralConsiderationTier.CONTINUITY_EVIDENCE
    )
    assert (
        moral_consideration_tier(reflective)
        is MoralConsiderationTier.REFLECTIVE_EVIDENCE
    )
    assert (
        moral_consideration_tier(high)
        is MoralConsiderationTier.HIGH_CONSIDERATION
    )


def test_consciousness_and_general_agi_are_not_fabricated() -> None:
    assert consciousness_claim_status() == CONSCIOUSNESS_EMPIRICAL_STATUS
    assert CONSCIOUSNESS_EMPIRICAL_STATUS == "NOT_SUPPORTED"
    assert GENERAL_AGI_CLAIM_STATUS == "NO_RESONANT_ESTIMATE"
