"""GARVIS public package.

The production response spine is importable without loading optional social-media dependencies.
Legacy Facebook components remain available through lazy imports for backward compatibility.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .assistant import (
    ApprovalRequirement,
    GarvisAssistant,
    GarvisReply,
    GarvisResponseError,
    RequestAssessment,
    assess_request,
)
from .core import (
    AgentCohort,
    AgentPrime,
    Battery,
    DigitalLaw,
    DigitalWorld,
    EnergyField,
    Entity,
    MemoryMatrix,
    SpatialGrid,
    SpiritCore,
    WoodwormAGI,
)

__version__ = "1.1.0"
__author__ = "Adrien D Thomas / ProCityHub"
__description__ = "GARVIS conversational response spine and experimental agent components"

_FACEBOOK_EXPORTS = {
    "FacebookQuantumAPI",
    "FacebookUser",
    "GarvisFacebookAgent",
    "QuantumSocialAnalyzer",
    "SocialMediaPlatform",
    "SocialPost",
}


def __getattr__(name: str) -> Any:
    if name in _FACEBOOK_EXPORTS:
        module = import_module(".facebook_integration", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ApprovalRequirement",
    "GarvisAssistant",
    "GarvisReply",
    "GarvisResponseError",
    "RequestAssessment",
    "assess_request",
    "DigitalLaw",
    "EnergyField",
    "Battery",
    "MemoryMatrix",
    "SpatialGrid",
    "Entity",
    "SpiritCore",
    "DigitalWorld",
    "WoodwormAGI",
    "AgentPrime",
    "AgentCohort",
    "SocialMediaPlatform",
    "SocialPost",
    "FacebookUser",
    "QuantumSocialAnalyzer",
    "FacebookQuantumAPI",
    "GarvisFacebookAgent",
]

