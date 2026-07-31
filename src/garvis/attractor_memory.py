"""Deterministic Hopfield-style associative memory.

This module demonstrates distributed pattern storage and bounded recall.
It does not establish consciousness, identity persistence, or biological memory.
"""

# Ruff's PEP 604 suggestions require Python 3.10 syntax.
# GARVIS supports Python 3.9, so Optional and Union remain intentional.
# ruff: noqa: UP007, UP045

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

BipolarPattern = tuple[int, ...]
WeightMatrix = tuple[tuple[float, ...], ...]
CorruptionAmount = Union[int, float]


@dataclass(frozen=True)
class RecallResult:
    """Structured result from one bounded recall attempt."""

    initial_state: BipolarPattern
    final_state: BipolarPattern
    converged: bool
    sweeps: int
    state_changes: int
    initial_energy: Optional[float]
    final_energy: float
    energy_trace: tuple[float, ...]
    exact_match: Optional[bool]
    hamming_distance: Optional[int]
    warnings: tuple[str, ...]


class HopfieldMemory:
    """Small deterministic asynchronous Hopfield network."""

    def __init__(self, patterns: Sequence[Sequence[int]]) -> None:
        normalized = tuple(self._validate_stored_patterns(patterns))
        self._patterns: tuple[BipolarPattern, ...] = normalized
        self.dimension = len(normalized[0])
        self.weights: WeightMatrix = self._build_weights(normalized)

    @staticmethod
    def _validate_integer_value(value: object, allowed: tuple[int, ...]) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"state values must be integers from {allowed}")
        if value not in allowed:
            raise ValueError(f"state values must be integers from {allowed}")
        return value

    @classmethod
    def _validate_stored_patterns(
        cls,
        patterns: Sequence[Sequence[int]],
    ) -> list[BipolarPattern]:
        if not patterns:
            raise ValueError("at least one stored pattern is required")

        normalized: list[BipolarPattern] = []
        expected_dimension: Optional[int] = None

        for pattern in patterns:
            values = tuple(cls._validate_integer_value(value, (-1, 1)) for value in pattern)

            if not values:
                raise ValueError("stored patterns cannot be empty")

            if expected_dimension is None:
                expected_dimension = len(values)
            elif len(values) != expected_dimension:
                raise ValueError("all stored patterns must have the same dimension")

            normalized.append(values)

        return normalized

    def _validate_cue(self, cue: Sequence[int]) -> BipolarPattern:
        values = tuple(self._validate_integer_value(value, (-1, 0, 1)) for value in cue)

        if len(values) != self.dimension:
            raise ValueError(
                f"cue dimension {len(values)} does not match network dimension {self.dimension}"
            )

        return values

    def _validate_target(self, target: Sequence[int]) -> BipolarPattern:
        values = tuple(self._validate_integer_value(value, (-1, 1)) for value in target)

        if len(values) != self.dimension:
            raise ValueError(
                f"target dimension {len(values)} does not match network dimension {self.dimension}"
            )

        return values

    @staticmethod
    def _build_weights(patterns: tuple[BipolarPattern, ...]) -> WeightMatrix:
        dimension = len(patterns[0])
        matrix: list[list[float]] = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

        for row in range(dimension):
            for column in range(dimension):
                if row == column:
                    matrix[row][column] = 0.0
                    continue

                total = sum(pattern[row] * pattern[column] for pattern in patterns)
                matrix[row][column] = total / dimension

        return tuple(tuple(row) for row in matrix)

    def energy(self, state: Sequence[int]) -> float:
        """Return the Hopfield energy of a complete bipolar state."""

        values = self._validate_target(state)
        total = 0.0

        for row in range(self.dimension):
            for column in range(self.dimension):
                total += values[row] * self.weights[row][column] * values[column]

        return -0.5 * total

    def recall(
        self,
        cue: Sequence[int],
        *,
        target: Optional[Sequence[int]] = None,
        max_sweeps: int = 50,
    ) -> RecallResult:
        """Settle a cue using deterministic asynchronous updates."""

        if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int):
            raise ValueError("max_sweeps must be a positive integer")
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be a positive integer")

        initial = self._validate_cue(cue)
        expected = self._validate_target(target) if target is not None else None
        state = list(initial)

        initial_energy = None
        if 0 not in state:
            initial_energy = self.energy(state)

        energy_trace: list[float] = []
        total_changes = 0
        converged = False
        energy_increased = False
        previous_energy = initial_energy
        completed_sweeps = 0

        for sweep in range(1, max_sweeps + 1):
            changes_this_sweep = 0

            for neuron in range(self.dimension):
                field = sum(
                    self.weights[neuron][other] * state[other] for other in range(self.dimension)
                )

                if field > 0:
                    new_value = 1
                elif field < 0:
                    new_value = -1
                elif state[neuron] in (-1, 1):
                    new_value = state[neuron]
                else:
                    new_value = 1

                if new_value != state[neuron]:
                    state[neuron] = new_value
                    changes_this_sweep += 1
                    total_changes += 1

            completed_sweeps = sweep
            current_energy = self.energy(state)
            energy_trace.append(current_energy)

            if previous_energy is not None and current_energy > previous_energy + 1e-12:
                energy_increased = True

            previous_energy = current_energy

            if changes_this_sweep == 0:
                converged = True
                break

        final_state = tuple(state)
        warnings: list[str] = []

        if not converged:
            warnings.append("maximum_sweeps_reached_without_convergence")

        if energy_increased:
            warnings.append("energy_increased_during_asynchronous_settling")

        exact_match: Optional[bool] = None
        hamming_distance: Optional[int] = None

        if expected is not None:
            hamming_distance = sum(
                actual != wanted for actual, wanted in zip(final_state, expected)
            )
            exact_match = hamming_distance == 0

            if converged and not exact_match:
                warnings.append("converged_to_non_target_attractor")

        return RecallResult(
            initial_state=initial,
            final_state=final_state,
            converged=converged,
            sweeps=completed_sweeps,
            state_changes=total_changes,
            initial_energy=initial_energy,
            final_energy=energy_trace[-1],
            energy_trace=tuple(energy_trace),
            exact_match=exact_match,
            hamming_distance=hamming_distance,
            warnings=tuple(warnings),
        )

    def nearest_pattern(self, cue: Sequence[int]) -> BipolarPattern:
        """Explicit-storage baseline using known-bit Hamming distance."""

        values = self._validate_cue(cue)
        known_positions = [index for index, value in enumerate(values) if value != 0]

        def distance(pattern: BipolarPattern) -> int:
            return sum(pattern[index] != values[index] for index in known_positions)

        return min(self._patterns, key=distance)

    def no_memory_baseline(self, cue: Sequence[int]) -> BipolarPattern:
        """Return the cue with erased values deterministically filled as +1."""

        values = self._validate_cue(cue)
        return tuple(1 if value == 0 else value for value in values)


def _resolve_corruption_count(
    amount: CorruptionAmount,
    dimension: int,
) -> int:
    if isinstance(amount, bool):
        raise ValueError("corruption amount must be an integer count or fraction")

    if isinstance(amount, int):
        count = amount
    elif isinstance(amount, float):
        if not 0.0 <= amount <= 1.0:
            raise ValueError("fractional corruption amount must be between 0 and 1")
        count = int((dimension * amount) + 0.5)
    else:
        raise ValueError("corruption amount must be an integer count or fraction")

    if not 0 <= count <= dimension:
        raise ValueError("corruption count exceeds pattern dimension")

    return count


def flip_bits(
    pattern: Sequence[int],
    amount: CorruptionAmount,
    *,
    seed: int,
) -> BipolarPattern:
    """Return a reproducibly bit-flipped copy of a bipolar pattern."""

    values = tuple(HopfieldMemory._validate_integer_value(value, (-1, 1)) for value in pattern)

    if not values:
        raise ValueError("pattern cannot be empty")

    count = _resolve_corruption_count(amount, len(values))
    positions = random.Random(seed).sample(range(len(values)), count)
    result = list(values)

    for position in positions:
        result[position] *= -1

    return tuple(result)


def erase_bits(
    pattern: Sequence[int],
    amount: CorruptionAmount,
    *,
    seed: int,
) -> BipolarPattern:
    """Return a reproducibly erased copy of a bipolar pattern."""

    values = tuple(HopfieldMemory._validate_integer_value(value, (-1, 1)) for value in pattern)

    if not values:
        raise ValueError("pattern cannot be empty")

    count = _resolve_corruption_count(amount, len(values))
    positions = random.Random(seed).sample(range(len(values)), count)
    result = list(values)

    for position in positions:
        result[position] = 0

    return tuple(result)
