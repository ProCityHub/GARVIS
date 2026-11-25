"""Hypercube Brain Engine — Adrien D. Thomas / ProCityHub.

Core truth/brain types plus sensory/motor pathway modules in this package.
"""

from .core import (
    ActionDecision,
    ActionProposal,
    HypercubeBrainEngine,
    HypercubeTruthEngine,
    Observation,
    TruthState,
    coherence,
)

__all__ = [
    "ActionDecision",
    "ActionProposal",
    "HypercubeBrainEngine",
    "HypercubeTruthEngine",
    "Observation",
    "TruthState",
    "coherence",
]
