"""Read-only probability metrics for GARVIS Hypercube Heartbeat hardware runs.

Creator / conceptual architect: Adrien D. Thomas (ProCityHub/GARVIS).

These functions compare observed and ideal classical measurement distributions.
They do not submit jobs, access credentials, infer consciousness, or turn a
hardware distribution into a truth/AGI score.

Python 3.9 compatible; standard library only.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple


class QuantumMetricError(ValueError):
    pass


def _validate_counts(counts: Mapping[str, int]) -> Tuple[int, int]:
    if not counts:
        raise QuantumMetricError("counts must not be empty")
    widths = {len(key) for key in counts}
    if len(widths) != 1:
        raise QuantumMetricError("count keys must use one bit width")
    width = next(iter(widths))
    if width <= 0:
        raise QuantumMetricError("bit width must be positive")
    total = 0
    for key, count in counts.items():
        if any(char not in "01" for char in key):
            raise QuantumMetricError("count keys must be bitstrings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise QuantumMetricError("counts must be non-negative integers")
        total += count
    if total <= 0:
        raise QuantumMetricError("total shots must be positive")
    return width, total


def normalize_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    _, total = _validate_counts(counts)
    return {key: count / float(total) for key, count in counts.items()}


def normalize_probabilities(probabilities: Mapping[str, float]) -> Dict[str, float]:
    if not probabilities:
        raise QuantumMetricError("probability mapping must not be empty")
    widths = {len(key) for key in probabilities}
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise QuantumMetricError("probability keys must use one positive bit width")
    total = 0.0
    values = {}
    for key, value in probabilities.items():
        if any(char not in "01" for char in key):
            raise QuantumMetricError("probability keys must be bitstrings")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise QuantumMetricError("probabilities must be finite and non-negative")
        values[key] = number
        total += number
    if total <= 0.0:
        raise QuantumMetricError("probability total must be positive")
    return {key: value / total for key, value in values.items()}


def reverse_bit_order(distribution: Mapping[str, float]) -> Dict[str, float]:
    """Reverse every displayed bitstring, preserving probability mass."""

    normalized = normalize_probabilities(distribution)
    return {key[::-1]: value for key, value in normalized.items()}


def shannon_entropy_bits(distribution: Mapping[str, float]) -> float:
    probs = normalize_probabilities(distribution)
    return -sum(value * math.log(value, 2) for value in probs.values() if value > 0.0)


def marginal_p1(distribution: Mapping[str, float], qubit_index: int) -> float:
    """P(q_i=1) assuming displayed strings are q[n-1]...q[0]."""

    probs = normalize_probabilities(distribution)
    width = len(next(iter(probs)))
    if isinstance(qubit_index, bool) or not isinstance(qubit_index, int) or not 0 <= qubit_index < width:
        raise QuantumMetricError("qubit_index outside distribution width")
    display_index = width - 1 - qubit_index
    return sum(value for key, value in probs.items() if key[display_index] == "1")


def bit_agreement(distribution: Mapping[str, float], left_qubit: int, right_qubit: int) -> float:
    probs = normalize_probabilities(distribution)
    width = len(next(iter(probs)))
    for index in (left_qubit, right_qubit):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width:
            raise QuantumMetricError("qubit index outside distribution width")
    left_display = width - 1 - left_qubit
    right_display = width - 1 - right_qubit
    return sum(value for key, value in probs.items() if key[left_display] == key[right_display])


def mutual_information_bits(distribution: Mapping[str, float], left_qubit: int, right_qubit: int) -> float:
    probs = normalize_probabilities(distribution)
    width = len(next(iter(probs)))
    for index in (left_qubit, right_qubit):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width:
            raise QuantumMetricError("qubit index outside distribution width")
    li = width - 1 - left_qubit
    ri = width - 1 - right_qubit
    joint = {(a, b): 0.0 for a in "01" for b in "01"}
    left = {"0": 0.0, "1": 0.0}
    right = {"0": 0.0, "1": 0.0}
    for key, value in probs.items():
        a, b = key[li], key[ri]
        joint[(a, b)] += value
        left[a] += value
        right[b] += value
    result = 0.0
    for (a, b), p_ab in joint.items():
        if p_ab > 0.0:
            result += p_ab * math.log(p_ab / (left[a] * right[b]), 2)
    return result


def _aligned(left: Mapping[str, float], right: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Tuple[str, ...]]:
    lp = normalize_probabilities(left)
    rp = normalize_probabilities(right)
    left_width = len(next(iter(lp)))
    right_width = len(next(iter(rp)))
    if left_width != right_width:
        raise QuantumMetricError("distributions must use the same bit width")
    keys = tuple(sorted(set(lp) | set(rp)))
    return lp, rp, keys


def bhattacharyya_coefficient(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    lp, rp, keys = _aligned(left, right)
    return sum(math.sqrt(lp.get(key, 0.0) * rp.get(key, 0.0)) for key in keys)


def classical_fidelity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    coefficient = bhattacharyya_coefficient(left, right)
    return coefficient * coefficient


def total_variation_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    lp, rp, keys = _aligned(left, right)
    return 0.5 * sum(abs(lp.get(key, 0.0) - rp.get(key, 0.0)) for key in keys)


def hellinger_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.sqrt(max(0.0, 1.0 - bhattacharyya_coefficient(left, right)))


def _kl_bits(left: Mapping[str, float], right: Mapping[str, float], keys: Sequence[str]) -> float:
    result = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        if p > 0.0:
            if q <= 0.0:
                return math.inf
            result += p * math.log(p / q, 2)
    return result


def jensen_shannon_divergence_bits(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    lp, rp, keys = _aligned(left, right)
    midpoint = {key: 0.5 * (lp.get(key, 0.0) + rp.get(key, 0.0)) for key in keys}
    return 0.5 * _kl_bits(lp, midpoint, keys) + 0.5 * _kl_bits(rp, midpoint, keys)


def top_outcomes(distribution: Mapping[str, float], limit: int = 10) -> Tuple[Tuple[str, float], ...]:
    probs = normalize_probabilities(distribution)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise QuantumMetricError("limit must be a positive integer")
    return tuple(sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:limit])


def compare_distributions(ideal: Mapping[str, float], observed: Mapping[str, float]) -> Dict[str, float]:
    """Robust classical comparison metrics; no single metric is a truth score."""

    return {
        "bhattacharyya_coefficient": bhattacharyya_coefficient(ideal, observed),
        "classical_fidelity": classical_fidelity(ideal, observed),
        "total_variation_distance": total_variation_distance(ideal, observed),
        "hellinger_distance": hellinger_distance(ideal, observed),
        "jensen_shannon_divergence_bits": jensen_shannon_divergence_bits(ideal, observed),
        "observed_entropy_bits": shannon_entropy_bits(observed),
    }


__all__ = [
    "QuantumMetricError",
    "bhattacharyya_coefficient",
    "bit_agreement",
    "classical_fidelity",
    "compare_distributions",
    "hellinger_distance",
    "jensen_shannon_divergence_bits",
    "marginal_p1",
    "mutual_information_bits",
    "normalize_counts",
    "normalize_probabilities",
    "reverse_bit_order",
    "shannon_entropy_bits",
    "top_outcomes",
    "total_variation_distance",
]
