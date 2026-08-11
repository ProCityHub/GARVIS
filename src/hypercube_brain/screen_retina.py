"""
GARVIS Android Screen Retina
Disposable research/prototype implementation.

Architecture / conceptual design:
Adrien D. Thomas
ProCityHub / GARVIS

Purpose
-------
Translate already-authorized Android screen/UI evidence into GARVIS's
existing VisualNervePacket contract.

One physical visible element becomes one vision evidence lineage.

Observation does not grant action authority.

This module DOES NOT:
- capture the Android display
- request Android permissions
- activate MediaProjection
- activate Accessibility
- click
- tap
- type
- send
- delete
- install
- deploy
- bypass Android security boundaries

O / A / B
---------
Observer:
    visible screen evidence

Bridge:
    screen geometry, source, confidence, visual field

Actor:
    later interpretation/proposed action

The Actor receives no execution authority from this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import hypot, isfinite

from .sensory_nervous_system import VisualNervePacket


def _finite(value: float, name: str) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def _clamp01(value: float) -> float:
    return max(
        0.0,
        min(
            1.0,
            _finite(value, "value"),
        ),
    )


@dataclass(frozen=True)
class ScreenBounds:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        left = _finite(
            self.left,
            "left",
        )
        top = _finite(
            self.top,
            "top",
        )
        right = _finite(
            self.right,
            "right",
        )
        bottom = _finite(
            self.bottom,
            "bottom",
        )

        if right <= left:
            raise ValueError(
                "right must be greater than left"
            )

        if bottom <= top:
            raise ValueError(
                "bottom must be greater than top"
            )


@dataclass(frozen=True)
class ScreenElement:
    element_id: str
    label: str
    bounds: ScreenBounds
    screen_width: float
    screen_height: float
    confidence: float = 1.0
    role: str = "unknown"
    motion: float = 0.0
    source: str = "android_screen"

    def __post_init__(self) -> None:
        if not self.element_id.strip():
            raise ValueError(
                "element_id required"
            )

        if _finite(
            self.screen_width,
            "screen_width",
        ) <= 0:
            raise ValueError(
                "screen_width must be > 0"
            )

        if _finite(
            self.screen_height,
            "screen_height",
        ) <= 0:
            raise ValueError(
                "screen_height must be > 0"
            )

        confidence = _finite(
            self.confidence,
            "confidence",
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be within [0,1]"
            )

        if not self.source.strip():
            raise ValueError(
                "source required"
            )

    @property
    def semantic_label(self) -> str:
        return (
            self.label.strip()
            or self.role.strip()
            or "unknown"
        )


@dataclass(frozen=True)
class RetinaCalibrationReport:
    rounds: int
    converged: bool
    stop_reason: str
    packet_count: int
    unique_lineages: int
    mean_confidence: float
    max_roundtrip_error: float
    provisional: bool = True

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)


class AndroidScreenRetina:
    """
    Map authorized Android screen semantics into GARVIS's
    existing visual nerve.

    The class has no execution methods.
    """

    source = "android_screen"

    @staticmethod
    def _field(
        x: float,
    ) -> str:
        if x < (1.0 / 3.0):
            return "left"

        if x > (2.0 / 3.0):
            return "right"

        return "central"

    def packet(
        self,
        element: ScreenElement,
    ) -> VisualNervePacket:

        center_x_pixels = (
            element.bounds.left
            + element.bounds.right
        ) / 2.0

        center_y_pixels = (
            element.bounds.top
            + element.bounds.bottom
        ) / 2.0

        x = _clamp01(
            center_x_pixels
            / element.screen_width
        )

        y = _clamp01(
            center_y_pixels
            / element.screen_height
        )

        return VisualNervePacket(
            label=element.semantic_label,
            x=x,
            y=y,
            depth=None,
            motion=_clamp01(
                element.motion
            ),
            confidence=_clamp01(
                element.confidence
            ),
            visual_field=self._field(x),
            source=element.source,
        )

    def packets(
        self,
        elements: Iterable[ScreenElement],
    ) -> tuple[
        VisualNervePacket,
        ...
    ]:
        """
        Exactly one output packet for each input element.

        This prevents the same physical observation from
        becoming artificial duplicate evidence.
        """

        return tuple(
            self.packet(element)
            for element in elements
        )

    def autocalibrate(
        self,
        elements: Iterable[ScreenElement],
        max_rounds: int = 9,
        stable_rounds: int = 2,
    ) -> RetinaCalibrationReport:
        """
        Bounded software-contract calibration.

        This is NOT empirical visual truth and does not
        claim biological vision.

        GARVIS repeatedly checks his screen-to-nerve
        transformation until the result is deterministic.

        Nothing is persisted by this function.
        """

        items = tuple(elements)

        if not items:
            raise ValueError(
                "at least one screen element is required"
            )

        if (
            max_rounds < 1
            or stable_rounds < 1
        ):
            raise ValueError(
                "round counts must be positive"
            )

        previous_signature = None
        stable = 0
        max_error = 0.0
        rounds = 0
        packets = ()

        for round_index in range(
            1,
            max_rounds + 1,
        ):
            packets = self.packets(
                items
            )

            signature = tuple(
                (
                    packet.label,
                    round(packet.x, 12),
                    round(packet.y, 12),
                    packet.visual_field,
                    round(
                        packet.motion,
                        12,
                    ),
                    round(
                        packet.confidence,
                        12,
                    ),
                    packet.source,
                )
                for packet in packets
            )

            for element, packet in zip(
                items,
                packets,
            ):
                expected_x = (
                    (
                        element.bounds.left
                        + element.bounds.right
                    )
                    / 2.0
                    / element.screen_width
                )

                expected_y = (
                    (
                        element.bounds.top
                        + element.bounds.bottom
                    )
                    / 2.0
                    / element.screen_height
                )

                error = hypot(
                    packet.x - expected_x,
                    packet.y - expected_y,
                )

                max_error = max(
                    max_error,
                    error,
                )

            rounds = round_index

            if (
                signature
                == previous_signature
            ):
                stable += 1
            else:
                stable = 0

            previous_signature = signature

            if stable >= stable_rounds:
                break

        converged = (
            stable
            >= stable_rounds
        )

        return RetinaCalibrationReport(
            rounds=rounds,
            converged=converged,
            stop_reason=(
                "deterministic_convergence"
                if converged
                else "max_rounds_reached"
            ),
            packet_count=len(
                packets
            ),
            unique_lineages=len(
                items
            ),
            mean_confidence=(
                sum(
                    packet.confidence
                    for packet
                    in packets
                )
                / len(packets)
            ),
            max_roundtrip_error=max_error,
        )


__all__ = [
    "ScreenBounds",
    "ScreenElement",
    "RetinaCalibrationReport",
    "AndroidScreenRetina",
]
