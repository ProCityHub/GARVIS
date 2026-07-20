"""Deterministic evidence-to-lattice-memory consolidation.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

This module converts a read-only Psychology Kernel EvidenceAssessment into a
bounded recurrent lattice-memory topology.

It does not establish truth, biological memory, consciousness, sentience,
quantum behavior, AGI, spiritual proof, or external-action authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from garvis.psychology.evidence_adapter import (
    EvidenceAssessment,
    EvidenceSignal,
)

from .pulse_bus import HeartbeatPulse
from .recurrent_memory import (
    MemoryAttractor,
    MirrorConnection,
    RecurrentLatticeMemory,
    RecurrentRecallResult,
)


def _unit(name: str, value: float) -> float:
    checked = float(value)

    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

    return checked


def _assessment_sha256(
    assessment: EvidenceAssessment,
) -> str:
    data = {
        "signals": [
            {
                "signal_id": signal.signal_id,
                "source_path": signal.source_path,
                "kind": signal.kind.value,
                "routing_weight": signal.routing_weight,
                "content_sha256": signal.content_sha256,
                "provenance_hash": signal.provenance_hash,
            }
            for signal in sorted(
                assessment.signals,
                key=lambda item: item.signal_id,
            )
        ],
        "evidence_sufficiency": assessment.evidence_sufficiency,
        "evidence_error": assessment.evidence_error,
        "unknown_fraction": assessment.unknown_fraction,
        "speculative_fraction": assessment.speculative_fraction,
    }

    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _signal_node(signal: EvidenceSignal) -> str:
    return (
        f"evidence:{signal.kind.value}:"
        f"{signal.signal_id[:20]}"
    )


@dataclass(frozen=True)
class ConsolidatedEvidenceMemory:
    """Immutable bridge between evidence signals and recurrent memory."""

    assessment_sha256: str
    center_node: str
    signal_nodes: Tuple[Tuple[str, str], ...]
    retained_signal_ids: Tuple[str, ...]
    excluded_signal_ids: Tuple[str, ...]
    memory: RecurrentLatticeMemory
    external_action_allowed: bool = False
    claim_boundary: str = (
        "Classical evidence-to-memory routing contract only. "
        "No truth guarantee, biological-memory, quantum, consciousness, "
        "sentience, AGI, spiritual-proof, or external-action claim."
    )

    def node_for_signal(
        self,
        signal_id: str,
    ) -> Optional[str]:
        for stored_signal_id, node_id in self.signal_nodes:
            if stored_signal_id == signal_id:
                return node_id

        return None

    def recall(
        self,
        signal_ids: Tuple[str, ...],
        pulse: HeartbeatPulse,
    ) -> RecurrentRecallResult:
        """Recall the consolidated attractor from selected evidence signals."""

        if not signal_ids:
            raise ValueError("signal_ids must not be empty")

        cue: dict[str, float] = {}

        for signal_id in signal_ids:
            node_id = self.node_for_signal(signal_id)

            if node_id is None:
                raise ValueError(
                    f"signal was not consolidated: {signal_id}"
                )

            cue[node_id] = 1.0

        return self.memory.recall(
            cue=cue,
            pulse=pulse,
        )


def consolidate_evidence_assessment(
    assessment: EvidenceAssessment,
    *,
    minimum_routing_weight: float = 0.25,
    max_signals: int = 32,
    recall_threshold: float = 0.70,
) -> ConsolidatedEvidenceMemory:
    """Convert accepted evidence signals into one recurrent attractor."""

    minimum_routing_weight = _unit(
        "minimum_routing_weight",
        minimum_routing_weight,
    )

    recall_threshold = _unit(
        "recall_threshold",
        recall_threshold,
    )

    if max_signals < 1:
        raise ValueError("max_signals must be at least one")

    ordered_signals = sorted(
        assessment.signals,
        key=lambda signal: (
            -signal.routing_weight,
            signal.kind.value,
            signal.signal_id,
        ),
    )

    retained = tuple(
        signal
        for signal in ordered_signals
        if signal.routing_weight > 0.0
        and signal.routing_weight >= minimum_routing_weight
    )[:max_signals]

    if not retained:
        raise ValueError(
            "assessment contains no signals eligible for consolidation"
        )

    retained_ids = {
        signal.signal_id
        for signal in retained
    }

    excluded_signal_ids = tuple(
        sorted(
            signal.signal_id
            for signal in assessment.signals
            if signal.signal_id not in retained_ids
        )
    )

    center_node = "evidence:center"

    signal_nodes = tuple(
        sorted(
            (
                signal.signal_id,
                _signal_node(signal),
            )
            for signal in retained
        )
    )

    nodes = (
        center_node,
        *tuple(
            node_id
            for _, node_id in signal_nodes
        ),
    )

    signal_by_id = {
        signal.signal_id: signal
        for signal in retained
    }

    mirrors = tuple(
        MirrorConnection(
            left=center_node,
            right=node_id,
            left_to_right=signal_by_id[
                signal_id
            ].routing_weight,
            right_to_left=signal_by_id[
                signal_id
            ].routing_weight,
        )
        for signal_id, node_id in signal_nodes
    )

    center_activation = max(
        0.01,
        assessment.evidence_sufficiency,
    )

    pattern = (
        (center_node, center_activation),
        *tuple(
            (
                node_id,
                signal_by_id[signal_id].routing_weight,
            )
            for signal_id, node_id in signal_nodes
        ),
    )

    provenance_refs = tuple(
        sorted(
            {
                signal.provenance_hash
                for signal in retained
            }
        )
    )

    assessment_hash = _assessment_sha256(assessment)

    attractor = MemoryAttractor(
        attractor_id=(
            f"evidence-attractor:{assessment_hash[:20]}"
        ),
        pattern=pattern,
        recall_threshold=recall_threshold,
        provenance_refs=provenance_refs,
    )

    inhibition = min(
        0.5,
        0.05
        + 0.25 * assessment.unknown_fraction
        + 0.25 * assessment.speculative_fraction,
    )

    memory = RecurrentLatticeMemory(
        nodes=nodes,
        mirrors=mirrors,
        attractors=(attractor,),
        persistence=0.4,
        inhibition=inhibition,
        convergence_tolerance=0.001,
        max_cycles=32,
    )

    return ConsolidatedEvidenceMemory(
        assessment_sha256=assessment_hash,
        center_node=center_node,
        signal_nodes=signal_nodes,
        retained_signal_ids=tuple(
            signal.signal_id
            for signal in retained
        ),
        excluded_signal_ids=excluded_signal_ids,
        memory=memory,
    )
