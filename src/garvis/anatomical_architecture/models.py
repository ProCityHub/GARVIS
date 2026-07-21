"""Typed models for the GARVIS anatomical architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrganSystem(str, Enum):
    INTEGUMENTARY = "integumentary"
    SKELETAL = "skeletal"
    MUSCULAR = "muscular"
    NERVOUS = "nervous"
    ENDOCRINE = "endocrine"
    CARDIOVASCULAR = "cardiovascular"
    LYMPHATIC_IMMUNE = "lymphatic_immune"
    RESPIRATORY = "respiratory"
    DIGESTIVE = "digestive"
    URINARY_EXCRETORY = "urinary_excretory"
    REPRODUCTIVE = "reproductive"


@dataclass(frozen=True)
class SystemDefinition:
    system: OrganSystem
    biological_role: str
    software_role: str
    responsibilities: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    risks: tuple[str, ...]


@dataclass(frozen=True)
class SystemSignal:
    source: OrganSystem
    target: OrganSystem
    kind: str
    payload: dict[str, Any]
    priority: float = 0.5


@dataclass(frozen=True)
class HeartbeatState:
    phase: str
    active_systems: tuple[OrganSystem, ...]
    observations: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
