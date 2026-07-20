"""Deterministic Hypercube Heartbeat pulse bus.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

The pulse bus synchronizes bounded classical state transitions. It performs no
network access, sensing, file writes, model calls, or outside-world execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from garvis.psychology import unified_center

from .resonance_packet import LatticeResonancePacket


def _unit(name: str, value: float) -> float:
    checked = float(value)

    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

    return checked


class PulsePhase(str, Enum):
    OBSERVE = "observe"
    ACTIVATE = "activate"
    PROPAGATE = "propagate"
    INHIBIT = "inhibit"
    INTEGRATE = "integrate"
    REST = "rest"


@dataclass(frozen=True)
class HeartbeatPulse:
    """One immutable Hypercube Heartbeat pulse."""

    cycle: int
    activation: float = 1.0
    wall_coherence: float = 0.6
    phase: PulsePhase = PulsePhase.ACTIVATE
    source_id: str = "hypercubeheartbeat"

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError("cycle must be zero or greater")

        _unit("activation", self.activation)
        _unit("wall_coherence", self.wall_coherence)

        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")

    @property
    def raw_union(self) -> float:
        """Preserve the arithmetic activation plus wall coherence."""

        return self.activation + self.wall_coherence

    @property
    def normalized_center(self) -> float:
        """Normalize the canonical 1.6 union into one full center unit."""

        return unified_center(
            self.activation,
            self.wall_coherence,
        )

    def compute_sha256(self) -> str:
        encoded = json.dumps(
            {
                "cycle": self.cycle,
                "activation": self.activation,
                "wall_coherence": self.wall_coherence,
                "phase": self.phase.value,
                "source_id": self.source_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class PulseResponse:
    """One deterministic pulse-to-memory result."""

    pulse_sha256: str
    packet_sha256_before: str
    packet_sha256_after: str
    cue_alignment: float
    retained_energy: float
    reflected_energy: float
    injected_energy: float
    energy_before_inhibition: float
    energy_after_inhibition: float
    recall_strength: float
    recalled: bool
    updated_packet: LatticeResonancePacket
    external_action_allowed: bool = False
    claim_boundary: str = (
        "Classical bounded associative-recall measurement only. "
        "No quantum, consciousness, biological-memory, AGI, spiritual-proof, "
        "or external-action claim."
    )


def calculate_cue_alignment(
    packet: LatticeResonancePacket,
    cue_weights: Mapping[str, float],
) -> float:
    """Measure overlap between a partial cue and a memory packet."""

    checked = {
        concept_id: _unit(
            f"cue weight for {concept_id}",
            value,
        )
        for concept_id, value in cue_weights.items()
    }

    return sum(
        component.amplitude
        * checked.get(component.concept_id, 0.0)
        for component in packet.normalized_components
    )


def apply_heartbeat(
    packet: LatticeResonancePacket,
    pulse: HeartbeatPulse,
    cue_weights: Mapping[str, float],
    reflection: float = 0.6,
    inhibition: float = 0.0,
    recall_threshold: float = 0.5,
) -> PulseResponse:
    """Apply one bounded heartbeat to one resonance packet."""

    reflection = _unit("reflection", reflection)
    inhibition = _unit("inhibition", inhibition)
    recall_threshold = _unit(
        "recall_threshold",
        recall_threshold,
    )

    if pulse.cycle < packet.cycle:
        raise ValueError("pulse cycle cannot precede packet cycle")

    cue_alignment = calculate_cue_alignment(
        packet,
        cue_weights,
    )

    retained_energy = (
        packet.energy
        * (1.0 - packet.decay_rate)
    )

    reflected_energy = (
        retained_energy
        * reflection
        * pulse.wall_coherence
    )

    injected_energy = (
        pulse.activation
        * pulse.normalized_center
        * cue_alignment
    )

    energy_before_inhibition = min(
        retained_energy
        + reflected_energy
        + injected_energy,
        1.0,
    )

    energy_after_inhibition = (
        energy_before_inhibition
        * (1.0 - inhibition)
    )

    recall_strength = (
        energy_after_inhibition
        * ((cue_alignment + packet.confidence) / 2.0)
        * (1.0 - packet.uncertainty)
    )

    updated_packet = packet.with_energy(
        cycle=max(packet.cycle + 1, pulse.cycle),
        energy=energy_after_inhibition,
    )

    return PulseResponse(
        pulse_sha256=pulse.compute_sha256(),
        packet_sha256_before=packet.compute_sha256(),
        packet_sha256_after=updated_packet.compute_sha256(),
        cue_alignment=cue_alignment,
        retained_energy=retained_energy,
        reflected_energy=reflected_energy,
        injected_energy=injected_energy,
        energy_before_inhibition=energy_before_inhibition,
        energy_after_inhibition=energy_after_inhibition,
        recall_strength=recall_strength,
        recalled=recall_strength >= recall_threshold,
        updated_packet=updated_packet,
    )


@dataclass(frozen=True)
class PulseBus:
    """Deterministic source and transmission boundary."""

    source_id: str = "hypercubeheartbeat"

    def emit(
        self,
        cycle: int,
        activation: float = 1.0,
        wall_coherence: float = 0.6,
        phase: PulsePhase = PulsePhase.ACTIVATE,
    ) -> HeartbeatPulse:
        return HeartbeatPulse(
            cycle=cycle,
            activation=activation,
            wall_coherence=wall_coherence,
            phase=phase,
            source_id=self.source_id,
        )

    def transmit(
        self,
        packet: LatticeResonancePacket,
        pulse: HeartbeatPulse,
        cue_weights: Mapping[str, float],
        reflection: float = 0.6,
        inhibition: float = 0.0,
        recall_threshold: float = 0.5,
    ) -> PulseResponse:
        if pulse.source_id != self.source_id:
            raise ValueError("pulse source does not match this bus")

        return apply_heartbeat(
            packet=packet,
            pulse=pulse,
            cue_weights=cue_weights,
            reflection=reflection,
            inhibition=inhibition,
            recall_threshold=recall_threshold,
        )
