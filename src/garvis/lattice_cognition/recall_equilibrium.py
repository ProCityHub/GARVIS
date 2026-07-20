"""Connect recurrent lattice recall to psychological equilibrium.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

This module evaluates whether a bounded recurrent-memory result is coherent
enough to support a proposal. It never grants external execution authority.

This is a classical engineering model, not biological memory, consciousness,
sentience, AGI, quantum behavior, spiritual proof, or clinical psychology.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from garvis.psychology.equilibrium import (
    SIX_WALL_COHERENCE,
    EquilibriumResult,
    evaluate_equilibrium,
)

from .recurrent_memory import RecurrentRecallResult


def _unit(name: str, value: float) -> float:
    checked = float(value)

    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

    return checked


def _recall_sha256(recall: RecurrentRecallResult) -> str:
    data = {
        "cycles_run": recall.cycles_run,
        "converged": recall.converged,
        "final_delta": recall.final_delta,
        "final_state": list(recall.final_state),
        "recalled_attractor_id": recall.recalled_attractor_id,
        "similarity": recall.similarity,
        "recalled": recall.recalled,
        "lattice_sha256": recall.lattice_sha256,
        "pulse_sha256": recall.pulse_sha256,
        "external_action_allowed": recall.external_action_allowed,
    }

    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecallEquilibriumAssessment:
    """Immutable evaluation of recall quality and equilibrium."""

    recall_sha256: str
    recall_quality: float
    convergence_quality: float
    evidence_quality: float
    action_quality: float
    context_quality: float
    constraints_passed: bool
    equilibrium: EquilibriumResult
    external_action_allowed: bool = False
    claim_boundary: str = (
        "Classical recall-equilibrium assessment only. "
        "No biological-memory, consciousness, sentience, AGI, quantum, "
        "spiritual-proof, clinical, or external-execution claim."
    )


def evaluate_recall_equilibrium(
    *,
    recall: RecurrentRecallResult,
    evidence_sufficiency: float,
    action_readiness: float,
    context_stability: float,
    constraints_passed: bool,
    external_action: bool,
    threshold: float = 0.95,
) -> RecallEquilibriumAssessment:
    """Convert a recall result into bounded equilibrium dimensions."""

    evidence_sufficiency = _unit(
        "evidence_sufficiency",
        evidence_sufficiency,
    )
    action_readiness = _unit(
        "action_readiness",
        action_readiness,
    )
    context_stability = _unit(
        "context_stability",
        context_stability,
    )
    threshold = _unit("threshold", threshold)
    similarity = _unit("recall similarity", recall.similarity)
    final_delta = _unit("recall final_delta", recall.final_delta)

    recall_quality = similarity if recall.recalled else 0.0

    convergence_quality = (
        1.0
        if recall.converged
        else max(0.0, 1.0 - final_delta)
    )

    evidence_quality = min(
        evidence_sufficiency,
        recall_quality,
    )

    action_quality = min(
        action_readiness,
        recall_quality,
        convergence_quality,
    )

    context_quality = min(
        context_stability,
        convergence_quality,
    )

    effective_constraints = bool(
        constraints_passed
        and recall.recalled
        and recall.converged
    )

    equilibrium = evaluate_equilibrium(
        activation=recall_quality,
        wall_coherence=(
            SIX_WALL_COHERENCE
            * convergence_quality
        ),
        evidence=evidence_quality,
        action_readiness=action_quality,
        context_stability=context_quality,
        tensions={
            "memory_conflict": 1.0 - recall_quality,
            "recall_uncertainty": 1.0 - evidence_quality,
            "convergence_error": 1.0 - convergence_quality,
        },
        constraints_passed=effective_constraints,
        external_action=external_action,
        threshold=threshold,
    )

    return RecallEquilibriumAssessment(
        recall_sha256=_recall_sha256(recall),
        recall_quality=recall_quality,
        convergence_quality=convergence_quality,
        evidence_quality=evidence_quality,
        action_quality=action_quality,
        context_quality=context_quality,
        constraints_passed=effective_constraints,
        equilibrium=equilibrium,
    )
