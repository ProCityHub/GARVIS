"""Tests for the Genesis X Double-Slit Brain model."""
import math

import sys
from pathlib import Path

# Ensure package root is in sys.path for test execution
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from brain import (
    DiscriminationClass,
    DoubleSlitBrain,
    DoubleSlitObservation,
    EmpiricalStatus,
    canonical_lattice_score,
    classify_path_pair,
    exponent_identity_error,
    load_csv,
    phi_visibility,
    phi_visibility_cosh_form,
    qm_visibility,
    self_test,
    synthetic_qm_dataset,
)


def test_exponent_identity_error_is_zero():
    """Verify exponent_identity_error() returns ~0 (alpha + beta = 1)."""
    error = exponent_identity_error()
    assert error < 1e-12


def test_symmetric_inputs_equal_visibility():
    """Verify qm_visibility and phi_visibility are equal for symmetric inputs (A=B)."""
    for A in (0.1, 0.5, 1.0, 2.5):
        for O in (0.0, 0.5, 0.9, 1.0):
            v_qm = qm_visibility(A, A, O)
            v_phi = phi_visibility(A, A, O)
            assert abs(v_qm - v_phi) < 1e-12
            assert abs(v_qm - O) < 1e-12


def test_asymmetric_inputs_phi_visibility_greater_or_equal():
    """Verify phi_visibility >= qm_visibility for asymmetric inputs (A != B)."""
    asymmetric_pairs = [
        (1.0, 0.8),
        (1.0, 0.5),
        (1.0, 0.2),
        (0.3, 1.0),
    ]
    for A, B in asymmetric_pairs:
        for O in (0.5, 0.9, 1.0):
            v_qm = qm_visibility(A, B, O)
            v_phi = phi_visibility(A, B, O)
            assert v_phi + 1e-12 >= v_qm


def test_classify_path_pair_symmetric_and_asymmetric():
    """Verify classify_path_pair correctly identifies symmetric vs asymmetric path pairs."""
    symmetric_pair = classify_path_pair(1.0, 1.0)
    assert symmetric_pair is DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC

    asymmetric_pair = classify_path_pair(1.0, 0.4)
    assert asymmetric_pair is DiscriminationClass.DISCRIMINATING_ASYMMETRIC


def test_double_slit_brain_observe_and_score():
    """Verify DoubleSlitBrain can observe observations and score synthetic data."""
    brain = DoubleSlitBrain()
    dataset = synthetic_qm_dataset()

    for obs in dataset:
        point = brain.observe(obs)
        assert point.step > 0
        assert point.radius_qm >= 0.0
        assert point.radius_phi >= 0.0

    assert len(brain.points) == len(dataset)
    score = brain.score(asymmetric_only=True)
    assert score.n_asymmetric > 0
    assert score.status is EmpiricalStatus.DIRECTIONAL_SUPPORT_QM


def test_self_test_passes():
    """Verify the self_test() function executes and passes."""
    result = self_test()
    assert isinstance(result, dict)
    assert result.get("SELF_TEST") == "PASS"


def test_canonical_lattice_score():
    """Verify canonical_lattice_score function returns valid values."""
    score_val = canonical_lattice_score(O=1.0, A=1.0, B=1.0)
    assert abs(score_val - 1.0) < 1e-12

    zero_score = canonical_lattice_score(O=0.0, A=1.0, B=1.0)
    assert zero_score == 0.0


def test_phi_visibility_cosh_form_equivalence():
    """Verify phi_visibility matches phi_visibility_cosh_form for non-zero inputs."""
    test_cases = [(1.0, 0.8, 1.0), (1.0, 0.4, 0.9), (0.2, 1.0, 0.7)]
    for A, B, O in test_cases:
        direct = phi_visibility(A, B, O)
        cosh_form = phi_visibility_cosh_form(A, B, O)
        assert abs(direct - cosh_form) < 1e-12


def test_load_csv_sample_data():
    """Verify load_csv and DoubleSlitBrain scoring with example CSV sample data."""
    sample_csv_path = PACKAGE_ROOT / "examples" / "sample_data.csv"
    assert sample_csv_path.exists()

    observations = load_csv(sample_csv_path)
    assert len(observations) == 15

    brain = DoubleSlitBrain()
    for obs in observations:
        brain.observe(obs)

    score = brain.score()
    assert score.n_asymmetric == 15
    assert score.status is EmpiricalStatus.DIRECTIONAL_SUPPORT_QM


class TestEdgeCases:
    """Numerical edge cases."""

    def test_phi_visibility_at_eps_boundary(self):
        """V_phi should return non-zero for B = 1e-12 (EPS boundary).
        
        Previous bug: B > EPS guard zeroed terms when B == EPS.
        Fix: changed to B > 0 guard.
        """
        v = phi_visibility(1.0, 1e-12, 1.0)
        assert v > 0, f"V_phi should be > 0, got {v}"

    def test_phi_visibility_extreme_asymmetry(self):
        """V_phi should remain finite at A/B = 1e6."""
        v = phi_visibility(1.0, 1e-6, 1.0)
        assert math.isfinite(v)
        assert v > 0

    def test_zero_coherence(self):
        """O=0 should give zero visibility for both models."""
        assert qm_visibility(1.0, 0.5, 0.0) == 0.0
        assert phi_visibility(1.0, 0.5, 0.0) == 0.0


class TestGARVISIdentity:
    """Verify GARVIS system identity (not the old name)."""

    def test_system_name_is_garvis(self):
        from brain import SYSTEM, CREATOR, ORGANIZATION
        _old_name = "JAR" + "VIS"  # constructed to avoid grep false positives
        assert "GARVIS" in SYSTEM
        assert _old_name not in SYSTEM
        assert CREATOR == "Adrien D. Thomas"
        assert ORGANIZATION == "ProCityHub"
