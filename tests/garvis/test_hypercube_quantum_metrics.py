import math

from garvis.hypercube_quantum_metrics import (
    bit_agreement,
    classical_fidelity,
    compare_distributions,
    jensen_shannon_divergence_bits,
    marginal_p1,
    mutual_information_bits,
    normalize_counts,
    reverse_bit_order,
    shannon_entropy_bits,
    total_variation_distance,
)


def test_normalize_counts_and_entropy():
    p = normalize_counts({"00": 2, "11": 2})
    assert p == {"00": 0.5, "11": 0.5}
    assert shannon_entropy_bits(p) == 1.0


def test_identical_distribution_metrics():
    p = {"00": 0.75, "11": 0.25}
    assert math.isclose(classical_fidelity(p, p), 1.0)
    assert total_variation_distance(p, p) == 0.0
    assert jensen_shannon_divergence_bits(p, p) == 0.0


def test_disjoint_distribution_metrics():
    p = {"00": 1.0}
    q = {"11": 1.0}
    assert classical_fidelity(p, q) == 0.0
    assert total_variation_distance(p, q) == 1.0
    assert math.isclose(jensen_shannon_divergence_bits(p, q), 1.0)


def test_qubit_orientation_helpers():
    p = {"001": 0.25, "101": 0.75}
    assert marginal_p1(p, 0) == 1.0
    assert marginal_p1(p, 2) == 0.75
    assert reverse_bit_order(p) == {"100": 0.25, "101": 0.75}


def test_agreement_and_mutual_information():
    p = {"00": 0.5, "11": 0.5}
    assert bit_agreement(p, 0, 1) == 1.0
    assert math.isclose(mutual_information_bits(p, 0, 1), 1.0)


def test_compare_distribution_keys():
    metrics = compare_distributions({"0": 0.8, "1": 0.2}, {"0": 0.7, "1": 0.3})
    assert set(metrics) == {
        "bhattacharyya_coefficient",
        "classical_fidelity",
        "total_variation_distance",
        "hellinger_distance",
        "jensen_shannon_divergence_bits",
        "observed_entropy_bits",
    }
