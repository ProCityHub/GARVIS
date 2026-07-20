"""Complete deterministic GARVIS lattice cognitive cycle.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

The cycle connects:

EvidenceEnvelope
    -> Psychology EvidenceAssessment
    -> recurrent lattice-memory consolidation
    -> Hypercube Heartbeat pulse
    -> associative recall
    -> Psychology Equilibrium evaluation
    -> bounded proposal status

The result never grants external execution authority.

This is a classical engineering model. It is not biological memory,
consciousness, sentience, AGI, quantum behavior, spiritual proof,
clinical psychology, or a truth guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from garvis.evidence_envelope import EvidenceEnvelope
from garvis.psychology.evidence_adapter import (
    EvidenceAssessment,
    adapt_evidence_envelope,
)

from .consolidation import (
    ConsolidatedEvidenceMemory,
    consolidate_evidence_assessment,
)
from .pulse_bus import (
    HeartbeatPulse,
    PulseBus,
    PulsePhase,
)
from .recall_equilibrium import (
    RecallEquilibriumAssessment,
    evaluate_recall_equilibrium,
)
from .recurrent_memory import RecurrentRecallResult


class CognitiveCycleStage(str, Enum):
    """Deterministic stages of one bounded cognitive cycle."""

    ADAPT_EVIDENCE = "adapt_evidence"
    CONSOLIDATE_MEMORY = "consolidate_memory"
    EMIT_HEARTBEAT = "emit_heartbeat"
    RECALL_ATTRACTOR = "recall_attractor"
    EVALUATE_EQUILIBRIUM = "evaluate_equilibrium"
    EVALUATE_PROPOSAL = "evaluate_proposal"


def _envelope_sha256(envelope: EvidenceEnvelope) -> str:
    return envelope.compute_sha256_hash(
        {
            "immutable_source_evidence": (
                envelope.immutable_source_evidence
            ),
            "hypercube_cycle_data": envelope.hypercube_cycle_data,
            "deterministic_agi_measurements": (
                envelope.deterministic_agi_measurements
            ),
            "scientific_claims": envelope.scientific_claims,
            "source_provenance_hashes": (
                envelope.source_provenance_hashes
            ),
            "approval_state": envelope.approval_state,
            "garvis_evidence_summary": (
                envelope.garvis_evidence_summary
            ),
            "engineering_inference": (
                envelope.engineering_inference
            ),
            "symbolic_interpretation": (
                envelope.symbolic_interpretation
            ),
            "unsupported_speculation": (
                envelope.unsupported_speculation
            ),
            "unknowns": envelope.unknowns,
        }
    )


def _cycle_sha256(
    *,
    cycle: int,
    envelope_sha256: str,
    consolidated: ConsolidatedEvidenceMemory,
    pulse: HeartbeatPulse,
    cue_signal_ids: Tuple[str, ...],
    recall_equilibrium: RecallEquilibriumAssessment,
    stages: Tuple[CognitiveCycleStage, ...],
) -> str:
    equilibrium = recall_equilibrium.equilibrium

    data = {
        "cycle": cycle,
        "envelope_sha256": envelope_sha256,
        "assessment_sha256": consolidated.assessment_sha256,
        "memory_sha256": consolidated.memory.compute_sha256(),
        "pulse_sha256": pulse.compute_sha256(),
        "cue_signal_ids": list(cue_signal_ids),
        "recall_sha256": recall_equilibrium.recall_sha256,
        "recall_quality": recall_equilibrium.recall_quality,
        "convergence_quality": (
            recall_equilibrium.convergence_quality
        ),
        "evidence_quality": recall_equilibrium.evidence_quality,
        "action_quality": recall_equilibrium.action_quality,
        "context_quality": recall_equilibrium.context_quality,
        "constraints_passed": (
            recall_equilibrium.constraints_passed
        ),
        "integrated_equilibrium": (
            equilibrium.integrated_equilibrium
        ),
        "equilibrium_reached": equilibrium.equilibrium_reached,
        "proposal_eligible": equilibrium.proposal_eligible,
        "human_approval_required": (
            equilibrium.human_approval_required
        ),
        "stages": [stage.value for stage in stages],
    }

    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LatticeCognitiveCycleResult:
    """Immutable result of one complete bounded cognitive cycle."""

    cycle: int
    envelope_sha256: str
    assessment: EvidenceAssessment
    consolidated_memory: ConsolidatedEvidenceMemory
    pulse: HeartbeatPulse
    cue_signal_ids: Tuple[str, ...]
    recall: RecurrentRecallResult
    recall_equilibrium: RecallEquilibriumAssessment
    completed_stages: Tuple[CognitiveCycleStage, ...]
    cycle_sha256: str
    external_action_allowed: bool = False
    claim_boundary: str = (
        "Classical deterministic cognitive-routing result only. "
        "No truth guarantee, biological-memory, consciousness, "
        "sentience, AGI, quantum, spiritual-proof, clinical, "
        "network, sensing, or external-execution claim."
    )

    @property
    def proposal_eligible(self) -> bool:
        return self.recall_equilibrium.equilibrium.proposal_eligible

    @property
    def human_approval_required(self) -> bool:
        return (
            self.recall_equilibrium
            .equilibrium
            .human_approval_required
        )

    @property
    def decision(self) -> str:
        if not self.proposal_eligible:
            return "BLOCKED"

        if self.human_approval_required:
            return "HUMAN_REVIEW_REQUIRED"

        return "INTERNAL_PROPOSAL_ELIGIBLE"


def run_lattice_cognitive_cycle(
    *,
    envelope: EvidenceEnvelope,
    cycle: int,
    cue_signal_ids: Optional[Tuple[str, ...]] = None,
    activation: float = 1.0,
    wall_coherence: float = 0.6,
    minimum_routing_weight: float = 0.25,
    max_signals: int = 32,
    recall_threshold: float = 0.70,
    action_readiness: float = 1.0,
    context_stability: float = 1.0,
    constraints_passed: bool = True,
    external_action: bool = False,
    equilibrium_threshold: float = 0.95,
    source_id: str = "hypercubeheartbeat",
) -> LatticeCognitiveCycleResult:
    """Run one complete deterministic lattice cognitive cycle."""

    if cycle < 0:
        raise ValueError("cycle must be zero or greater")

    assessment = adapt_evidence_envelope(envelope)

    consolidated = consolidate_evidence_assessment(
        assessment,
        minimum_routing_weight=minimum_routing_weight,
        max_signals=max_signals,
        recall_threshold=recall_threshold,
    )

    selected_cues = (
        consolidated.retained_signal_ids
        if cue_signal_ids is None
        else tuple(cue_signal_ids)
    )

    if not selected_cues:
        raise ValueError("cue_signal_ids must not be empty")

    if len(set(selected_cues)) != len(selected_cues):
        raise ValueError("cue_signal_ids must be unique")

    pulse = PulseBus(source_id=source_id).emit(
        cycle=cycle,
        activation=activation,
        wall_coherence=wall_coherence,
        phase=PulsePhase.ACTIVATE,
    )

    recall = consolidated.recall(
        signal_ids=selected_cues,
        pulse=pulse,
    )

    recall_equilibrium = evaluate_recall_equilibrium(
        recall=recall,
        evidence_sufficiency=assessment.evidence_sufficiency,
        action_readiness=action_readiness,
        context_stability=context_stability,
        constraints_passed=constraints_passed,
        external_action=external_action,
        threshold=equilibrium_threshold,
    )

    completed_stages = (
        CognitiveCycleStage.ADAPT_EVIDENCE,
        CognitiveCycleStage.CONSOLIDATE_MEMORY,
        CognitiveCycleStage.EMIT_HEARTBEAT,
        CognitiveCycleStage.RECALL_ATTRACTOR,
        CognitiveCycleStage.EVALUATE_EQUILIBRIUM,
        CognitiveCycleStage.EVALUATE_PROPOSAL,
    )

    envelope_hash = _envelope_sha256(envelope)

    return LatticeCognitiveCycleResult(
        cycle=cycle,
        envelope_sha256=envelope_hash,
        assessment=assessment,
        consolidated_memory=consolidated,
        pulse=pulse,
        cue_signal_ids=selected_cues,
        recall=recall,
        recall_equilibrium=recall_equilibrium,
        completed_stages=completed_stages,
        cycle_sha256=_cycle_sha256(
            cycle=cycle,
            envelope_sha256=envelope_hash,
            consolidated=consolidated,
            pulse=pulse,
            cue_signal_ids=selected_cues,
            recall_equilibrium=recall_equilibrium,
            stages=completed_stages,
        ),
    )
