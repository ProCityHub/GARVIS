"""Classical Lattice Resonance Packet contracts.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

A packet is a bounded distributed representation used for associative recall.
It is not quantum superposition, biological memory, consciousness, or AGI.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Tuple


def _unit(name: str, value: float) -> float:
    checked = float(value)

    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

    return checked


def _nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


class PermissionClass(str, Enum):
    """Maximum authority represented by a packet."""

    LOCAL_REASONING = "local_reasoning"
    READ_ONLY = "read_only"
    PROPOSAL_ONLY = "proposal_only"


@dataclass(frozen=True)
class ResonanceComponent:
    """One concept participating in a classical weighted mixture."""

    concept_id: str
    amplitude: float
    phase_position: float = 0.0
    associations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty("concept_id", self.concept_id)

        if float(self.amplitude) < 0.0:
            raise ValueError("amplitude must not be negative")

        _unit("phase_position", self.phase_position)

        for association in self.associations:
            _nonempty("association", association)


@dataclass(frozen=True)
class LatticeResonancePacket:
    """Immutable pulse-addressable associative-memory packet."""

    packet_id: str
    cycle: int
    components: Tuple[ResonanceComponent, ...]
    energy: float
    confidence: float
    uncertainty: float
    decay_rate: float
    provenance_refs: Tuple[str, ...] = ()
    permission_class: PermissionClass = PermissionClass.LOCAL_REASONING
    claim_boundary: str = (
        "Classical distributed representation only. "
        "Not quantum superposition, consciousness, sentience, AGI, "
        "biological memory, spiritual proof, or execution authority."
    )

    def __post_init__(self) -> None:
        _nonempty("packet_id", self.packet_id)
        _nonempty("claim_boundary", self.claim_boundary)

        if self.cycle < 0:
            raise ValueError("cycle must be zero or greater")

        if not self.components:
            raise ValueError("components must not be empty")

        concept_ids = [
            component.concept_id
            for component in self.components
        ]

        if len(set(concept_ids)) != len(concept_ids):
            raise ValueError("component concept IDs must be unique")

        if sum(component.amplitude for component in self.components) <= 0.0:
            raise ValueError("component amplitudes require a positive total")

        _unit("energy", self.energy)
        _unit("confidence", self.confidence)
        _unit("uncertainty", self.uncertainty)
        _unit("decay_rate", self.decay_rate)

        for reference in self.provenance_refs:
            _nonempty("provenance reference", reference)

    @property
    def normalized_components(self) -> Tuple[ResonanceComponent, ...]:
        """Return a deterministic mixture whose amplitudes sum to one."""

        total = sum(
            component.amplitude
            for component in self.components
        )

        return tuple(
            ResonanceComponent(
                concept_id=component.concept_id,
                amplitude=component.amplitude / total,
                phase_position=component.phase_position,
                associations=tuple(sorted(component.associations)),
            )
            for component in sorted(
                self.components,
                key=lambda item: item.concept_id,
            )
        )

    @property
    def dominant_component(self) -> ResonanceComponent:
        """Return the strongest currently represented concept."""

        return max(
            self.normalized_components,
            key=lambda component: component.amplitude,
        )

    @property
    def effective_component_count(self) -> float:
        """Measure how broadly activation is distributed."""

        concentration = sum(
            component.amplitude ** 2
            for component in self.normalized_components
        )

        return 1.0 / concentration

    @property
    def entropy(self) -> float:
        """Return Shannon entropy of the classical concept mixture."""

        return -sum(
            component.amplitude * math.log(component.amplitude)
            for component in self.normalized_components
            if component.amplitude > 0.0
        )

    def decayed(self, cycles: int = 1) -> "LatticeResonancePacket":
        """Return a new packet after bounded passive decay."""

        if cycles < 0:
            raise ValueError("cycles must not be negative")

        return replace(
            self,
            cycle=self.cycle + cycles,
            energy=self.energy * ((1.0 - self.decay_rate) ** cycles),
        )

    def with_energy(
        self,
        cycle: int,
        energy: float,
    ) -> "LatticeResonancePacket":
        """Return a new packet after a heartbeat changes its energy."""

        if cycle < self.cycle:
            raise ValueError("cycle cannot move backwards")

        return replace(
            self,
            cycle=cycle,
            energy=_unit("energy", energy),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible packet data."""

        return {
            "packet_id": self.packet_id,
            "cycle": self.cycle,
            "components": [
                {
                    "concept_id": component.concept_id,
                    "amplitude": component.amplitude,
                    "phase_position": component.phase_position,
                    "associations": list(component.associations),
                }
                for component in self.normalized_components
            ],
            "energy": self.energy,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "decay_rate": self.decay_rate,
            "provenance_refs": sorted(self.provenance_refs),
            "permission_class": self.permission_class.value,
            "claim_boundary": self.claim_boundary,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def compute_sha256(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()

    def find_component(
        self,
        concept_id: str,
    ) -> Optional[ResonanceComponent]:
        for component in self.normalized_components:
            if component.concept_id == concept_id:
                return component

        return None
