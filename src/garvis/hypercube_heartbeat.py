"""Adrien D. Thomas Hypercube Heartbeat cognitive pulse.

A state-driven GARVIS engineering model.

The Heartbeat determines cognitive state transitions and release readiness.
Wall-clock watchdogs detect stalled computation only; they do not define
thought completion.

This is not a claim that biological brains literally implement these
equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from garvis.memory_lifecycle import (
    EvidenceStatus,
    MemoryRecord,
    retention_score,
)


PHI = (1.0 + math.sqrt(5.0)) / 2.0
TAU = 2.0 * math.pi


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Perspective(str, Enum):
    LITERAL = "000"
    CONTEXT = "001"
    INTENT = "010"
    RELATION = "011"
    EVIDENCE = "100"
    POSSIBILITY = "101"
    CONSEQUENCE = "110"
    INTEGRATION = "111"


PERSPECTIVES = (
    Perspective.LITERAL,
    Perspective.CONTEXT,
    Perspective.INTENT,
    Perspective.RELATION,
    Perspective.EVIDENCE,
    Perspective.POSSIBILITY,
    Perspective.CONSEQUENCE,
    Perspective.INTEGRATION,
)


class MemoryField(str, Enum):
    ACTIVE = "active"
    PERIPHERAL = "peripheral"
    CONSOLIDATED = "consolidated"


def oab_coherence(observer: float, actor: float, background: float) -> float:
    """Canonical Adrien D. Thomas relationship.

    C = O^1 * A^(1/phi) * B^(1/phi^2)
    """
    o = _clamp(observer)
    a = _clamp(actor)
    b = _clamp(background)

    return _clamp(
        o
        * a ** (1.0 / PHI)
        * b ** (1.0 / (PHI * PHI))
    )


def perspective_for_theta(theta: float) -> Perspective:
    theta = float(theta) % TAU
    sector = int(theta / (TAU / 8.0)) % 8
    return PERSPECTIVES[sector]


def evidence_quality(status: EvidenceStatus) -> float:
    return {
        EvidenceStatus.VERIFIED: 1.00,
        EvidenceStatus.EVIDENCE_SUPPORTED: 0.82,
        EvidenceStatus.USER_SUPPLIED: 0.62,
        EvidenceStatus.PROVISIONAL: 0.35,
        EvidenceStatus.MODEL_GENERATED: 0.12,
    }[status]


@dataclass(frozen=True)
class SemanticNode:
    key: str
    content: str

    weight: float = 0.0
    relation: float = 0.0
    evidence: float = 0.0
    goal: float = 0.0
    consequence: float = 0.0
    novelty: float = 0.0
    retention: float = 1.0

    repetition: int = 1
    protected: bool = False

    def pulse(self) -> "SemanticNode":
        """Advance one semantic-charge update.

        Repetition is deliberately absent. Repetition may reinforce memory
        retention, but repetition alone is never evidence.
        """
        weight = (
            0.55 * _clamp(self.retention) * _clamp(self.weight)
            + 0.10 * _clamp(self.relation)
            + 0.15 * _clamp(self.evidence)
            + 0.10 * _clamp(self.goal)
            + 0.05 * _clamp(self.consequence)
            + 0.05 * _clamp(self.novelty)
        )
        return replace(self, weight=_clamp(weight))


@dataclass(frozen=True)
class PulseMetrics:
    observer: float = 0.0
    actor: float = 0.0
    background: float = 0.0

    load: float = 0.0
    prediction_error: float = 0.0
    uncertainty: float = 0.0
    goal_urgency: float = 0.0
    meaningful_change: float = 0.0
    contradiction: float = 0.0

    evidence_quality: float = 0.0
    task_completion: float = 0.0

    @property
    def coherence(self) -> float:
        return oab_coherence(
            self.observer,
            self.actor,
            self.background,
        )

    @property
    def pressure(self) -> float:
        return _clamp(
            0.20 * self.load
            + 0.25 * self.prediction_error
            + 0.20 * self.uncertainty
            + 0.15 * self.goal_urgency
            + 0.20 * self.meaningful_change
        )

    @property
    def release_readiness(self) -> float:
        return _clamp(
            self.coherence
            * _clamp(self.evidence_quality)
            * (1.0 - _clamp(self.uncertainty))
            * _clamp(self.task_completion)
        )


@dataclass(frozen=True)
class EventBoundary:
    triggered: bool
    reason: str = ""


def detect_event_boundary(
    metrics: PulseMetrics,
    *,
    contradiction_threshold: float = 0.55,
    prediction_error_threshold: float = 0.65,
    change_threshold: float = 0.75,
) -> EventBoundary:
    if metrics.contradiction >= contradiction_threshold:
        return EventBoundary(True, "contradiction")

    if metrics.prediction_error >= prediction_error_threshold:
        return EventBoundary(True, "prediction_error")

    if metrics.meaningful_change >= change_threshold:
        return EventBoundary(True, "meaningful_change")

    return EventBoundary(False)


@dataclass(frozen=True)
class HeartbeatState:
    theta: float = 0.0
    revolutions: int = 0
    perspective: Perspective = Perspective.LITERAL
    last_pressure: float = 0.0

    def advance(
        self,
        metrics: PulseMetrics,
        *,
        dt: float = 1.0,
        omega_min: float = math.pi / 16.0,
        omega_max: float = math.pi,
    ) -> "HeartbeatState":
        pressure = metrics.pressure

        sigmoid = 1.0 / (
            1.0 + math.exp(-8.0 * (pressure - 0.5))
        )

        omega = (
            omega_min
            + (omega_max - omega_min) * sigmoid
        )

        raw = self.theta + max(0.0, float(dt)) * omega

        revolutions = self.revolutions + int(raw // TAU)
        theta = raw % TAU

        return HeartbeatState(
            theta=theta,
            revolutions=revolutions,
            perspective=perspective_for_theta(theta),
            last_pressure=pressure,
        )


@dataclass(frozen=True)
class ReleaseGate:
    threshold: float = 0.62

    def ready(
        self,
        metrics: PulseMetrics,
        *,
        deterministic_gates_passed: bool,
    ) -> bool:
        return (
            deterministic_gates_passed
            and metrics.release_readiness >= self.threshold
        )


@dataclass(frozen=True)
class AdaptiveWatchdog:
    """Process-stall watchdog. Never a thought-completion clock."""

    minimum_seconds: float = 15.0
    maximum_seconds: float = 900.0
    safety_margin: float = 1.8

    def estimate(
        self,
        *,
        prompt_tokens: int,
        requested_output_tokens: int,
        observed_tokens_per_second: float,
        device_load: float = 0.0,
        recent_runtime_seconds: float = 0.0,
    ) -> float:
        tokens = max(
            0,
            int(prompt_tokens)
            + int(requested_output_tokens),
        )

        throughput = max(
            0.1,
            float(observed_tokens_per_second),
        )

        base = tokens / throughput

        budget = (
            base
            * self.safety_margin
            * (1.0 + _clamp(device_load))
            + max(0.0, recent_runtime_seconds) * 0.25
        )

        return max(
            self.minimum_seconds,
            min(self.maximum_seconds, budget),
        )


def node_from_memory(
    memory: MemoryRecord,
    *,
    relevance: float,
) -> SemanticNode:
    retention = retention_score(memory)

    weight = _clamp(
        0.45 * _clamp(relevance)
        + 0.30 * retention
        + 0.15 * memory.salience
        + 0.10 * memory.confidence
    )

    return SemanticNode(
        key=f"memory:{memory.id}",
        content=memory.content or memory.trace_hint,
        weight=weight,
        relation=_clamp(relevance),
        evidence=evidence_quality(memory.evidence_status),
        goal=_clamp(memory.salience),
        consequence=_clamp(memory.arousal),
        retention=retention,
        repetition=max(1, memory.repetition_count),
        protected=memory.protected,
    )


def classify_field(
    node: SemanticNode,
    *,
    active_threshold: float = 0.55,
    peripheral_threshold: float = 0.18,
) -> MemoryField | None:
    if node.weight >= active_threshold:
        return MemoryField.ACTIVE

    if node.protected:
        return MemoryField.CONSOLIDATED

    if node.weight >= peripheral_threshold:
        return MemoryField.PERIPHERAL

    if (
        node.evidence
        >= evidence_quality(
            EvidenceStatus.EVIDENCE_SUPPORTED
        )
    ):
        return MemoryField.CONSOLIDATED

    return None


def prune_field(
    nodes: Iterable[SemanticNode],
) -> dict[MemoryField, tuple[SemanticNode, ...]]:
    fields: dict[MemoryField, list[SemanticNode]] = {
        MemoryField.ACTIVE: [],
        MemoryField.PERIPHERAL: [],
        MemoryField.CONSOLIDATED: [],
    }

    for node in nodes:
        field = classify_field(node)

        if field is not None:
            fields[field].append(node)

    return {
        field: tuple(
            sorted(
                values,
                key=lambda node: (-node.weight, node.key),
            )
        )
        for field, values in fields.items()
    }
