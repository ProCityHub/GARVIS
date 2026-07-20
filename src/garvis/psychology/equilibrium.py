"""Lattice-Psychology equilibrium mathematics by Adrien D. Thomas."""

from dataclasses import dataclass
from math import prod
from typing import Mapping

FULL_ACTIVATION = 1.0
SIX_WALL_COHERENCE = 0.6
UNIFIED_SCALE = 1.6


def unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def unified_center(activation: float, wall_coherence: float) -> float:
    """Preserve 1.0 + 0.6 = 1.6, then normalize 1.6 to one full unit."""
    activation = unit("activation", activation)
    wall_coherence = unit("wall_coherence", wall_coherence)
    return min((activation + wall_coherence) / UNIFIED_SCALE, 1.0)


def psychological_coherence(tensions: Mapping[str, float]) -> tuple[float, str | None]:
    """Convert normalized unresolved tensions into coherence."""
    if not tensions:
        return 1.0, None

    checked = {name: unit(name, value) for name, value in tensions.items()}
    limiting = max(checked, key=checked.get)
    disequilibrium = sum(checked.values()) / len(checked)
    return 1.0 - disequilibrium, limiting


def geometric_equilibrium(dimensions: Mapping[str, float]) -> float:
    """Integrate required dimensions without letting one strength hide a failure."""
    if not dimensions:
        return 1.0

    values = [unit(name, value) for name, value in dimensions.items()]
    return prod(values) ** (1.0 / len(values))


@dataclass(frozen=True)
class EquilibriumResult:
    raw_union: float
    normalized_center: float
    psychological_coherence: float
    integrated_equilibrium: float
    equilibrium_error: float
    limiting_dimension: str | None
    corner_bits: tuple[int, int, int]
    corner_index: int
    equilibrium_reached: bool
    proposal_eligible: bool
    human_approval_required: bool


def evaluate_equilibrium(
    *,
    activation: float,
    wall_coherence: float,
    evidence: float,
    action_readiness: float,
    context_stability: float,
    tensions: Mapping[str, float],
    constraints_passed: bool,
    external_action: bool,
    threshold: float = 0.95,
) -> EquilibriumResult:
    activation = unit("activation", activation)
    wall_coherence = unit("wall_coherence", wall_coherence)
    evidence = unit("evidence", evidence)
    action_readiness = unit("action_readiness", action_readiness)
    context_stability = unit("context_stability", context_stability)
    threshold = unit("threshold", threshold)

    center = unified_center(activation, wall_coherence)
    psyche, limiting = psychological_coherence(tensions)

    integrated = geometric_equilibrium(
        {
            "center": center,
            "psychology": psyche,
            "evidence": evidence,
            "action": action_readiness,
            "context": context_stability,
        }
    )

    corner_bits = (
        int(evidence >= threshold),
        int(action_readiness >= threshold),
        int(context_stability >= threshold),
    )
    corner_index = corner_bits[0] * 4 + corner_bits[1] * 2 + corner_bits[2]

    reached = bool(constraints_passed and integrated >= threshold)
    eligible = bool(reached and corner_bits == (1, 1, 1))

    return EquilibriumResult(
        raw_union=activation + wall_coherence,
        normalized_center=center,
        psychological_coherence=psyche,
        integrated_equilibrium=integrated,
        equilibrium_error=1.0 - integrated,
        limiting_dimension=limiting,
        corner_bits=corner_bits,
        corner_index=corner_index,
        equilibrium_reached=reached,
        proposal_eligible=eligible,
        human_approval_required=bool(external_action),
    )
