"""Deterministic recurrent two-way-mirror lattice memory.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

Memory is modeled as a learned recurrent topology capable of reconstructing
a distributed attractor from a partial cue.

This is a bounded classical engineering model. It is not biological memory,
quantum superposition, consciousness, sentience, AGI, or spiritual proof.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .pulse_bus import HeartbeatPulse


def _nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _unit(name: str, value: float) -> float:
    checked = float(value)

    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

    return checked


def _positive_unit(name: str, value: float) -> float:
    checked = _unit(name, value)

    if checked == 0.0:
        raise ValueError(f"{name} must be greater than zero")

    return checked


@dataclass(frozen=True)
class MirrorConnection:
    """A reciprocal connection between two lattice nodes."""

    left: str
    right: str
    left_to_right: float
    right_to_left: float

    def __post_init__(self) -> None:
        _nonempty("left", self.left)
        _nonempty("right", self.right)

        if self.left == self.right:
            raise ValueError("a mirror cannot connect a node to itself")

        _positive_unit("left_to_right", self.left_to_right)
        _positive_unit("right_to_left", self.right_to_left)

    @property
    def canonical_pair(self) -> Tuple[str, str]:
        return tuple(sorted((self.left, self.right)))  # type: ignore[return-value]


@dataclass(frozen=True)
class MemoryAttractor:
    """One distributed pattern that may be reconstructed by resonance."""

    attractor_id: str
    pattern: Tuple[Tuple[str, float], ...]
    recall_threshold: float = 0.75
    provenance_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty("attractor_id", self.attractor_id)

        if not self.pattern:
            raise ValueError("pattern must not be empty")

        node_ids = [node_id for node_id, _ in self.pattern]

        if len(set(node_ids)) != len(node_ids):
            raise ValueError("pattern node IDs must be unique")

        if not any(value > 0.0 for _, value in self.pattern):
            raise ValueError("pattern requires positive activation")

        for node_id, value in self.pattern:
            _nonempty("pattern node", node_id)
            _unit(f"pattern value for {node_id}", value)

        _unit("recall_threshold", self.recall_threshold)

        for reference in self.provenance_refs:
            _nonempty("provenance reference", reference)

    @property
    def canonical_pattern(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted(self.pattern))


@dataclass(frozen=True)
class RecurrentRecallResult:
    """Deterministic result of recurrent lattice recall."""

    cycles_run: int
    converged: bool
    final_delta: float
    final_state: Tuple[Tuple[str, float], ...]
    recalled_attractor_id: Optional[str]
    similarity: float
    recalled: bool
    lattice_sha256: str
    pulse_sha256: str
    external_action_allowed: bool = False
    claim_boundary: str = (
        "Classical recurrent associative-memory result only. "
        "No biological-memory, quantum, consciousness, sentience, AGI, "
        "spiritual-proof, or external-action claim."
    )

    def state_dict(self) -> dict[str, float]:
        return dict(self.final_state)


@dataclass(frozen=True)
class RecurrentLatticeMemory:
    """Immutable recurrent topology with reciprocal mirror connections."""

    nodes: Tuple[str, ...]
    mirrors: Tuple[MirrorConnection, ...]
    attractors: Tuple[MemoryAttractor, ...]
    persistence: float = 0.4
    inhibition: float = 0.05
    convergence_tolerance: float = 0.001
    max_cycles: int = 32

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("nodes must not be empty")

        for node in self.nodes:
            _nonempty("node", node)

        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique")

        _unit("persistence", self.persistence)
        _unit("inhibition", self.inhibition)
        _positive_unit(
            "convergence_tolerance",
            self.convergence_tolerance,
        )

        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least one")

        known_nodes = set(self.nodes)
        seen_pairs: set[Tuple[str, str]] = set()

        for mirror in self.mirrors:
            if mirror.left not in known_nodes:
                raise ValueError(f"unknown mirror node: {mirror.left}")

            if mirror.right not in known_nodes:
                raise ValueError(f"unknown mirror node: {mirror.right}")

            if mirror.canonical_pair in seen_pairs:
                raise ValueError(
                    f"duplicate mirror pair: {mirror.canonical_pair}"
                )

            seen_pairs.add(mirror.canonical_pair)

        attractor_ids = [
            attractor.attractor_id
            for attractor in self.attractors
        ]

        if len(set(attractor_ids)) != len(attractor_ids):
            raise ValueError("attractor IDs must be unique")

        for attractor in self.attractors:
            for node_id, _ in attractor.pattern:
                if node_id not in known_nodes:
                    raise ValueError(
                        f"unknown attractor node: {node_id}"
                    )

    def canonical_json(self) -> str:
        data = {
            "nodes": sorted(self.nodes),
            "mirrors": [
                {
                    "left": mirror.left,
                    "right": mirror.right,
                    "left_to_right": mirror.left_to_right,
                    "right_to_left": mirror.right_to_left,
                }
                for mirror in sorted(
                    self.mirrors,
                    key=lambda item: item.canonical_pair,
                )
            ],
            "attractors": [
                {
                    "attractor_id": attractor.attractor_id,
                    "pattern": list(attractor.canonical_pattern),
                    "recall_threshold": attractor.recall_threshold,
                    "provenance_refs": sorted(
                        attractor.provenance_refs
                    ),
                }
                for attractor in sorted(
                    self.attractors,
                    key=lambda item: item.attractor_id,
                )
            ],
            "persistence": self.persistence,
            "inhibition": self.inhibition,
            "convergence_tolerance": self.convergence_tolerance,
            "max_cycles": self.max_cycles,
        }

        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def compute_sha256(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()

    def _checked_vector(
        self,
        name: str,
        values: Mapping[str, float],
    ) -> dict[str, float]:
        unknown = set(values) - set(self.nodes)

        if unknown:
            raise ValueError(
                f"{name} contains unknown nodes: {sorted(unknown)}"
            )

        return {
            node: _unit(
                f"{name} value for {node}",
                values.get(node, 0.0),
            )
            for node in self.nodes
        }

    def _reflected_signal(
        self,
        node: str,
        state: Mapping[str, float],
    ) -> float:
        weighted_sum = 0.0
        total_weight = 0.0

        for mirror in self.mirrors:
            if node == mirror.right:
                weighted_sum += (
                    state[mirror.left]
                    * mirror.left_to_right
                )
                total_weight += mirror.left_to_right

            elif node == mirror.left:
                weighted_sum += (
                    state[mirror.right]
                    * mirror.right_to_left
                )
                total_weight += mirror.right_to_left

        return weighted_sum / max(1.0, total_weight)

    def step(
        self,
        state: Mapping[str, float],
        cue: Mapping[str, float],
        pulse: HeartbeatPulse,
    ) -> dict[str, float]:
        """Run one bounded pulse through the reciprocal topology."""

        checked_state = self._checked_vector("state", state)
        checked_cue = self._checked_vector("cue", cue)

        global_activity = (
            sum(checked_state.values())
            / len(self.nodes)
        )

        next_state: dict[str, float] = {}

        for node in self.nodes:
            reflected = self._reflected_signal(
                node,
                checked_state,
            )

            raw_activation = (
                self.persistence * checked_state[node]
                + pulse.wall_coherence * reflected
                + pulse.activation * checked_cue[node]
                - self.inhibition * global_activity
            )

            next_state[node] = min(
                1.0,
                max(0.0, raw_activation),
            )

        return next_state

    def _similarity(
        self,
        state: Mapping[str, float],
        attractor: MemoryAttractor,
    ) -> float:
        target = dict(attractor.canonical_pattern)

        state_norm = math.sqrt(
            sum(state[node] ** 2 for node in self.nodes)
        )

        target_norm = math.sqrt(
            sum(target.get(node, 0.0) ** 2 for node in self.nodes)
        )

        if state_norm == 0.0 or target_norm == 0.0:
            return 0.0

        dot_product = sum(
            state[node] * target.get(node, 0.0)
            for node in self.nodes
        )

        return min(
            1.0,
            max(0.0, dot_product / (state_norm * target_norm)),
        )

    def recall(
        self,
        cue: Mapping[str, float],
        pulse: HeartbeatPulse,
    ) -> RecurrentRecallResult:
        """Reconstruct the closest learned attractor from a partial cue."""

        checked_cue = self._checked_vector("cue", cue)

        if not any(value > 0.0 for value in checked_cue.values()):
            raise ValueError("cue requires positive activation")

        state = {node: 0.0 for node in self.nodes}
        converged = False
        final_delta = 1.0
        cycles_run = 0

        for cycle in range(1, self.max_cycles + 1):
            next_state = self.step(
                state=state,
                cue=checked_cue,
                pulse=pulse,
            )

            final_delta = max(
                abs(next_state[node] - state[node])
                for node in self.nodes
            )

            state = next_state
            cycles_run = cycle

            if final_delta <= self.convergence_tolerance:
                converged = True
                break

        best_attractor: Optional[MemoryAttractor] = None
        best_similarity = 0.0

        for attractor in sorted(
            self.attractors,
            key=lambda item: item.attractor_id,
        ):
            similarity = self._similarity(
                state,
                attractor,
            )

            if similarity > best_similarity:
                best_attractor = attractor
                best_similarity = similarity

        recalled = (
            best_attractor is not None
            and best_similarity >= best_attractor.recall_threshold
        )

        return RecurrentRecallResult(
            cycles_run=cycles_run,
            converged=converged,
            final_delta=final_delta,
            final_state=tuple(sorted(state.items())),
            recalled_attractor_id=(
                best_attractor.attractor_id
                if recalled and best_attractor is not None
                else None
            ),
            similarity=best_similarity,
            recalled=recalled,
            lattice_sha256=self.compute_sha256(),
            pulse_sha256=pulse.compute_sha256(),
        )
