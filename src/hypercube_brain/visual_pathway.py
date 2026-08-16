"""Visual-pathway organ for the Hypercube Brain Engine.

Creator: Adrien D. Thomas / ProCityHub

Scientific inspiration:
human retina -> optic nerve -> chiasm -> tract -> LGN ->
optic radiations -> visual cortex.

This is a functional software analogy, not a biological simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .core import Observation, clamp01


@dataclass(frozen=True)
class VisualFeature:
    """One spatially located visual feature."""

    feature_id: str
    label: str
    x: float
    y: float
    confidence: float

    visual_field: str

    intensity: float = 0.0
    motion: float = 0.0
    detail: float = 0.0
    color_signal: float = 0.0
    depth: float | None = None

    source: str = "synthetic_camera"

    def __post_init__(self):
        if self.visual_field not in {"left", "right", "central"}:
            raise ValueError("visual_field must be left, right, or central")

        if not 0.0 <= self.x <= 1.0:
            raise ValueError("x must be between 0 and 1")

        if not 0.0 <= self.y <= 1.0:
            raise ValueError("y must be between 0 and 1")


@dataclass(frozen=True)
class OpticPacket:
    """Spatial packet analogous to routed retinal output."""

    feature_id: str
    label: str

    x: float
    y: float

    visual_field: str
    cortical_hemisphere: str

    confidence: float
    intensity: float
    motion: float
    detail: float
    color_signal: float
    depth: float | None

    source: str


@dataclass(frozen=True)
class LGNChannels:
    """Engineering parallel channels inspired by visual-pathway segregation."""

    packet: OpticPacket

    temporal_motion: float
    spatial_detail: float
    color_detail: float


@dataclass(frozen=True)
class CorticalPercept:
    """Integrated spatial percept for higher cognition."""

    feature_id: str
    label: str

    hemisphere: str
    visual_field: str

    x: float
    y: float
    depth: float | None

    motion_strength: float
    detail_strength: float
    color_strength: float

    confidence: float
    source: str


@dataclass(frozen=True)
class CollateralSignals:
    """Fast side-channel signals inspired by subcortical collateral pathways."""

    orienting_attention: bool
    bright_light_response: bool
    environmental_light_signal: float


class RetinaEncoder:
    """Normalize raw candidate features while preserving spatial coordinates."""

    def encode(self, features: Iterable[VisualFeature]) -> tuple[VisualFeature, ...]:
        output = []

        for feature in features:
            output.append(
                VisualFeature(
                    feature_id=feature.feature_id,
                    label=feature.label,
                    x=feature.x,
                    y=feature.y,
                    confidence=clamp01(feature.confidence),
                    visual_field=feature.visual_field,
                    intensity=clamp01(feature.intensity),
                    motion=clamp01(feature.motion),
                    detail=clamp01(feature.detail),
                    color_signal=clamp01(feature.color_signal),
                    depth=feature.depth,
                    source=feature.source,
                )
            )

        return tuple(output)


class OpticChiasmRouter:
    """Map visual-field information to contralateral cortical processing."""

    def route(self, feature: VisualFeature) -> OpticPacket:
        if feature.visual_field == "left":
            hemisphere = "right"
        elif feature.visual_field == "right":
            hemisphere = "left"
        else:
            hemisphere = "bilateral"

        return OpticPacket(
            feature_id=feature.feature_id,
            label=feature.label,
            x=feature.x,
            y=feature.y,
            visual_field=feature.visual_field,
            cortical_hemisphere=hemisphere,
            confidence=feature.confidence,
            intensity=feature.intensity,
            motion=feature.motion,
            detail=feature.detail,
            color_signal=feature.color_signal,
            depth=feature.depth,
            source=feature.source,
        )


class LGNGate:
    """Build bounded parallel feature channels before cortical integration."""

    def gate(self, packet: OpticPacket) -> LGNChannels:
        return LGNChannels(
            packet=packet,
            temporal_motion=clamp01(packet.motion),
            spatial_detail=clamp01(packet.detail),
            color_detail=clamp01(packet.color_signal),
        )


class VisualCortexMap:
    """Integrate routed channels while preserving retinotopic coordinates."""

    def integrate(self, channels: LGNChannels) -> CorticalPercept:
        packet = channels.packet

        return CorticalPercept(
            feature_id=packet.feature_id,
            label=packet.label,
            hemisphere=packet.cortical_hemisphere,
            visual_field=packet.visual_field,
            x=packet.x,
            y=packet.y,
            depth=packet.depth,
            motion_strength=channels.temporal_motion,
            detail_strength=channels.spatial_detail,
            color_strength=channels.color_detail,
            confidence=packet.confidence,
            source=packet.source,
        )


class CollateralRouter:
    """Generate fast orienting/light signals without declaring semantic truth."""

    def route(self, feature: VisualFeature) -> CollateralSignals:
        return CollateralSignals(
            orienting_attention=feature.motion >= 0.70,
            bright_light_response=feature.intensity >= 0.80,
            environmental_light_signal=feature.intensity,
        )


class VisualPathway:
    """Full software pathway from visual feature to cortical percept."""

    def __init__(self) -> None:
        self.retina = RetinaEncoder()
        self.chiasm = OpticChiasmRouter()
        self.lgn = LGNGate()
        self.cortex = VisualCortexMap()
        self.collateral = CollateralRouter()

    def process(
        self,
        features: Iterable[VisualFeature],
    ) -> tuple[tuple[CorticalPercept, ...], tuple[CollateralSignals, ...]]:

        retinal = self.retina.encode(features)

        percepts = []
        collateral = []

        for feature in retinal:
            packet = self.chiasm.route(feature)
            channels = self.lgn.gate(packet)
            percepts.append(self.cortex.integrate(channels))
            collateral.append(self.collateral.route(feature))

        return tuple(percepts), tuple(collateral)

    @staticmethod
    def to_brain_observation(percept: CorticalPercept) -> Observation:
        """Translate perception into an evidence object for the Truth Machine."""

        depth = (
            "unknown"
            if percept.depth is None
            else f"{percept.depth:.2f}"
        )

        content = (
            f"{percept.label}; "
            f"field={percept.visual_field}; "
            f"x={percept.x:.3f}; "
            f"y={percept.y:.3f}; "
            f"depth={depth}; "
            f"motion={percept.motion_strength:.3f}"
        )

        return Observation(
            content=content,
            source=f"visual_pathway:{percept.source}",
            confidence=percept.confidence,
            independent_group="vision",
        )

    @staticmethod
    def describe(percept: CorticalPercept) -> dict:
        return asdict(percept)
