"""GARVIS embodied sensory nervous system.

Creator: Adrien D. Thomas / ProCityHub

Functional analogies:

camera -> retina
visual packets -> optic nerve
microphone -> cochlear input
audio packets -> auditory nerve
router -> thalamic relay
multimodal integration -> association cortex
LLM bridge -> language/association cortex
phone telemetry -> body-state representation

These are engineering analogies, not biological equivalence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt

from .core import HypercubeBrainEngine, Observation, clamp01


class BinarySignal:
    """Explicit bridge between bytes and underlying binary representation."""

    @staticmethod
    def bits(data: bytes) -> str:
        return "".join(f"{value:08b}" for value in data)

    @staticmethod
    def bytes_from_bits(bits: str) -> bytes:
        if len(bits) % 8:
            raise ValueError("binary signal length must be divisible by 8")

        if set(bits) - {"0", "1"}:
            raise ValueError("binary signal may contain only 0 and 1")

        return bytes(
            int(bits[index:index + 8], 2)
            for index in range(0, len(bits), 8)
        )


@dataclass(frozen=True)
class VisualNervePacket:
    label: str
    x: float
    y: float
    depth: float | None
    motion: float
    confidence: float
    visual_field: str
    source: str = "camera"

    def observation(self) -> Observation:
        depth = "unknown" if self.depth is None else f"{self.depth:.2f}"

        return Observation(
            content=(
                f"visual:{self.label};"
                f"field={self.visual_field};"
                f"x={self.x:.3f};"
                f"y={self.y:.3f};"
                f"depth={depth};"
                f"motion={self.motion:.3f}"
            ),
            source=f"visual_nerve:{self.source}",
            confidence=clamp01(self.confidence),
            independent_group="vision",
        )


@dataclass(frozen=True)
class AuditoryNervePacket:
    rms_energy: float
    zero_crossing_rate: float
    dominant_band: str
    speech_probability: float
    confidence: float
    channel: str = "mono"
    source: str = "microphone"

    def observation(self) -> Observation:
        return Observation(
            content=(
                f"audio:band={self.dominant_band};"
                f"rms={self.rms_energy:.4f};"
                f"zcr={self.zero_crossing_rate:.4f};"
                f"speech_probability={self.speech_probability:.3f};"
                f"channel={self.channel}"
            ),
            source=f"auditory_nerve:{self.source}",
            confidence=clamp01(self.confidence),
            independent_group="audition",
        )


@dataclass(frozen=True)
class BodyStatePacket:
    battery: float
    thermal_pressure: float
    network_available: bool
    microphone_enabled: bool
    camera_enabled: bool

    def observation(self) -> Observation:
        return Observation(
            content=(
                f"body:battery={self.battery:.3f};"
                f"thermal={self.thermal_pressure:.3f};"
                f"network={self.network_available};"
                f"camera={self.camera_enabled};"
                f"microphone={self.microphone_enabled}"
            ),
            source="phone_body",
            confidence=1.0,
            independent_group="body",
        )


class DigitalCochlea:
    """Small dependency-free signal encoder for prototype PCM frames."""

    @staticmethod
    def encode(samples: Iterable[float]) -> AuditoryNervePacket:
        values = [max(-1.0, min(1.0, float(v))) for v in samples]

        if not values:
            raise ValueError("audio frame cannot be empty")

        rms = sqrt(sum(v * v for v in values) / len(values))

        crossings = sum(
            1
            for left, right in zip(values, values[1:])
            if (left < 0 <= right) or (left >= 0 > right)
        )

        zcr = crossings / max(1, len(values) - 1)

        if zcr < 0.08:
            band = "low"
        elif zcr < 0.24:
            band = "mid"
        else:
            band = "high"

        # Prototype feature only. This is not a speech recognition model.
        speech_probability = clamp01(
            0.30
            + min(rms, 0.40)
            + (0.20 if 0.03 <= zcr <= 0.30 else 0.0)
        )

        return AuditoryNervePacket(
            rms_energy=rms,
            zero_crossing_rate=zcr,
            dominant_band=band,
            speech_probability=speech_probability,
            confidence=0.75,
        )


class ThalamicRouter:
    """Route sensory packets without deciding semantic truth."""

    @staticmethod
    def route(packet):
        if isinstance(packet, VisualNervePacket):
            return "visual_cortex", packet.observation()

        if isinstance(packet, AuditoryNervePacket):
            return "auditory_cortex", packet.observation()

        if isinstance(packet, BodyStatePacket):
            return "somatic_body_map", packet.observation()

        raise TypeError(f"unsupported sensory packet: {type(packet).__name__}")


class MultimodalAssociationCortex:
    """Combine modalities while preserving separate provenance."""

    def integrate(self, packets) -> tuple[Observation, ...]:
        observations = []

        for packet in packets:
            _, observation = ThalamicRouter.route(packet)
            observations.append(observation)

        return tuple(observations)


class LanguageCortexBridge:
    """Translate multimodal state into bounded text for an LLM/reasoning organ."""

    @staticmethod
    def render(observations: Iterable[Observation]) -> str:
        lines = ["MULTIMODAL_SENSORY_STATE"]

        for observation in observations:
            lines.append(
                f"- source={observation.source} "
                f"confidence={observation.confidence:.3f} "
                f"content={observation.content}"
            )

        return "\n".join(lines)


class GarvisEmbodiedBrain:
    """Connect the phone-body sensory map to the Hypercube Brain heartbeat."""

    def __init__(self, brain: HypercubeBrainEngine | None = None) -> None:
        self.brain = brain or HypercubeBrainEngine()
        self.association = MultimodalAssociationCortex()

    def perceive(
        self,
        *,
        claim: str,
        packets,
        background: float = 1.0,
    ):
        observations = list(self.association.integrate(packets))

        return self.brain.heartbeat(
            claim=claim,
            observations=observations,
            background=background,
        )


def activation_policy(sensor: str) -> str:
    """Live sensors remain explicit governed boundaries."""

    normalized = sensor.strip().lower()

    if normalized in {"camera", "microphone", "screen"}:
        return "APPROVAL_REQUIRED_FOR_LIVE_SENSOR_ACTIVATION"

    return "UNKNOWN_SENSOR_FAIL_CLOSED"
