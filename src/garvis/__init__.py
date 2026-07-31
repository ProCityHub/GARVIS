<<<<<<< HEAD
"""
GARVIS - Pro Sync AGI with Quantum Consciousness
Advanced AI system with Facebook integration and consciousness simulation
"""

=======
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
>>>>>>> origin/main
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
<<<<<<< HEAD
from .facebook_integration import (
    FacebookQuantumAPI,
    FacebookUser,
    GarvisFacebookAgent,
    QuantumSocialAnalyzer,
    SocialMediaPlatform,
    SocialPost,
)

__version__ = "1.0.0"
__author__ = "ProCityHub"
__description__ = "GARVIS - Quantum consciousness AGI with social media integration"

__all__ = [
    # Core components
    'DigitalLaw',
    'EnergyField',
    'Battery',
    'MemoryMatrix',
    'SpatialGrid',
    'Entity',
    'SpiritCore',
    'DigitalWorld',
    'WoodwormAGI',
    'AgentPrime',
    'AgentCohort',

    # Facebook integration
    'SocialMediaPlatform',
    'SocialPost',
    'FacebookUser',
    'QuantumSocialAnalyzer',
    'FacebookQuantumAPI',
    'GarvisFacebookAgent'
=======

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
>>>>>>> origin/main
]

