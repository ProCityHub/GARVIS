"""Hypercube heartbeat across the 11 anatomy-inspired systems."""

from __future__ import annotations

from dataclasses import dataclass

from .models import HeartbeatState, OrganSystem, SystemSignal
from .registry import SYSTEMS


@dataclass(frozen=True)
class HeartbeatResult:
    states: tuple[HeartbeatState, ...]
    routed_signals: tuple[SystemSignal, ...]
    summary: str


class AnatomicalHeartbeat:
    """Coordinate the 11 systems through 0.0 → 0.6 → 1.0 → 1.6."""

    def run(self, request: str) -> HeartbeatResult:
        clean = request.strip()
        if not clean:
            raise ValueError("request must not be empty")

        states = (
            HeartbeatState(
                phase="0.0",
                active_systems=(
                    OrganSystem.INTEGUMENTARY,
                    OrganSystem.RESPIRATORY,
                    OrganSystem.DIGESTIVE,
                ),
                observations=(
                    "capture the exact input",
                    "validate the boundary",
                    "measure resource pressure",
                    "segment and normalize content",
                ),
            ),
            HeartbeatState(
                phase="0.6",
                active_systems=(
                    OrganSystem.NERVOUS,
                    OrganSystem.ENDOCRINE,
                    OrganSystem.LYMPHATIC_IMMUNE,
                    OrganSystem.SKELETAL,
                ),
                observations=(
                    "retrieve relevant memory",
                    "apply policy and authority",
                    "check risk and integrity",
                    "verify schemas and contracts",
                ),
            ),
            HeartbeatState(
                phase="1.0",
                active_systems=(
                    OrganSystem.NERVOUS,
                    OrganSystem.MUSCULAR,
                    OrganSystem.CARDIOVASCULAR,
                    OrganSystem.RESPIRATORY,
                ),
                observations=(
                    "form the response or plan",
                    "route signals",
                    "execute only permitted actions",
                    "monitor output flow",
                ),
            ),
            HeartbeatState(
                phase="1.6",
                active_systems=(
                    OrganSystem.URINARY_EXCRETORY,
                    OrganSystem.DIGESTIVE,
                    OrganSystem.REPRODUCTIVE,
                    OrganSystem.NERVOUS,
                ),
                observations=(
                    "archive the complete episode",
                    "remove duplicate working data",
                    "consolidate durable knowledge",
                    "propose reusable modules or tests",
                ),
            ),
        )

        signals = (
            SystemSignal(
                source=OrganSystem.INTEGUMENTARY,
                target=OrganSystem.DIGESTIVE,
                kind="validated_input",
                payload={"request": clean},
                priority=1.0,
            ),
            SystemSignal(
                source=OrganSystem.DIGESTIVE,
                target=OrganSystem.NERVOUS,
                kind="normalized_content",
                payload={"request": clean},
                priority=0.9,
            ),
            SystemSignal(
                source=OrganSystem.NERVOUS,
                target=OrganSystem.LYMPHATIC_IMMUNE,
                kind="risk_check",
                payload={"request": clean},
                priority=0.9,
            ),
            SystemSignal(
                source=OrganSystem.NERVOUS,
                target=OrganSystem.MUSCULAR,
                kind="execution_plan",
                payload={"mode": "answer_or_prepare"},
                priority=0.8,
            ),
            SystemSignal(
                source=OrganSystem.MUSCULAR,
                target=OrganSystem.CARDIOVASCULAR,
                kind="execution_result",
                payload={"status": "ready_for_delivery"},
                priority=0.8,
            ),
            SystemSignal(
                source=OrganSystem.NERVOUS,
                target=OrganSystem.URINARY_EXCRETORY,
                kind="context_cleanup",
                payload={"preserve_archive": True},
                priority=0.7,
            ),
        )

        summary = (
            "GARVIS anatomical heartbeat: 0.0 receives and digests the signal; "
            "0.6 checks coherence, policy, structure, and immune risk; "
            "1.0 plans, routes, and executes bounded output; "
            "1.6 archives, cleans working state, consolidates knowledge, "
            "and proposes controlled new modules."
        )
        return HeartbeatResult(states=states, routed_signals=signals, summary=summary)
