from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from garvis.hypercube_heartbeat import (
    AdaptiveWatchdog,
    HeartbeatState,
    MemoryField,
    PHI,
    PERSPECTIVES,
    PulseMetrics,
    ReleaseGate,
    SemanticNode,
    classify_field,
    detect_event_boundary,
    evidence_quality,
    node_from_memory,
    oab_coherence,
    perspective_for_theta,
)
from garvis.memory_lifecycle import (
    EvidenceStatus,
    MemoryKind,
    MemoryRecord,
    MemoryState,
)


def test_oab_canonical_relationship() -> None:
    o, a, b = 0.9, 0.8, 0.7

    expected = (
        o
        * a ** (1.0 / PHI)
        * b ** (1.0 / (PHI * PHI))
    )

    assert oab_coherence(o, a, b) == pytest.approx(
        expected
    )


def test_eight_phase_circle_order() -> None:
    step = 2.0 * math.pi / 8.0

    observed = tuple(
        perspective_for_theta(
            (index + 0.25) * step
        )
        for index in range(8)
    )

    assert observed == PERSPECTIVES


def test_111_to_000_creates_revolution() -> None:
    state = HeartbeatState(
        theta=2.0 * math.pi - 0.01,
        perspective=PERSPECTIVES[-1],
    )

    result = state.advance(
        PulseMetrics(
            load=1.0,
            prediction_error=1.0,
            uncertainty=1.0,
            goal_urgency=1.0,
            meaningful_change=1.0,
        ),
        dt=0.2,
    )

    assert result.revolutions == 1


def test_repetition_is_not_evidence() -> None:
    one = SemanticNode(
        key="one",
        content="claim",
        weight=0.2,
        evidence=0.1,
        repetition=1,
    ).pulse()

    repeated = SemanticNode(
        key="many",
        content="claim claim claim",
        weight=0.2,
        evidence=0.1,
        repetition=1000,
    ).pulse()

    assert one.evidence == repeated.evidence
    assert one.weight == pytest.approx(repeated.weight)


def test_verified_outranks_model_generated_evidence() -> None:
    assert (
        evidence_quality(EvidenceStatus.VERIFIED)
        >
        evidence_quality(EvidenceStatus.MODEL_GENERATED)
    )


def test_prediction_error_creates_boundary() -> None:
    boundary = detect_event_boundary(
        PulseMetrics(prediction_error=0.90)
    )

    assert boundary.triggered
    assert boundary.reason == "prediction_error"


def test_contradiction_takes_boundary_priority() -> None:
    boundary = detect_event_boundary(
        PulseMetrics(
            contradiction=0.9,
            prediction_error=0.9,
        )
    )

    assert boundary.reason == "contradiction"


def test_uncertainty_suppresses_release() -> None:
    certain = PulseMetrics(
        observer=1,
        actor=1,
        background=1,
        evidence_quality=1,
        uncertainty=0,
        task_completion=1,
    )

    uncertain = PulseMetrics(
        observer=1,
        actor=1,
        background=1,
        evidence_quality=1,
        uncertainty=0.9,
        task_completion=1,
    )

    assert (
        certain.release_readiness
        >
        uncertain.release_readiness
    )


def test_release_cannot_bypass_deterministic_gate() -> None:
    metrics = PulseMetrics(
        observer=1,
        actor=1,
        background=1,
        evidence_quality=1,
        task_completion=1,
    )

    gate = ReleaseGate()

    assert gate.ready(
        metrics,
        deterministic_gates_passed=True,
    )

    assert not gate.ready(
        metrics,
        deterministic_gates_passed=False,
    )


def test_protected_low_weight_information_consolidates() -> None:
    node = SemanticNode(
        key="protected",
        content="protected evidence",
        weight=0.01,
        protected=True,
    )

    assert (
        classify_field(node)
        == MemoryField.CONSOLIDATED
    )


def test_weak_unsupported_noise_can_prune() -> None:
    node = SemanticNode(
        key="noise",
        content="noise",
        weight=0.01,
        evidence=0.0,
    )

    assert classify_field(node) is None


def test_evidence_can_survive_below_active_threshold() -> None:
    node = SemanticNode(
        key="evidence",
        content="supported observation",
        weight=0.05,
        evidence=evidence_quality(
            EvidenceStatus.EVIDENCE_SUPPORTED
        ),
    )

    assert (
        classify_field(node)
        == MemoryField.CONSOLIDATED
    )


def test_high_pressure_accelerates_pulse() -> None:
    low = HeartbeatState().advance(
        PulseMetrics(),
        dt=0.1,
    )

    high = HeartbeatState().advance(
        PulseMetrics(
            load=1,
            prediction_error=1,
            uncertainty=1,
            goal_urgency=1,
            meaningful_change=1,
        ),
        dt=0.1,
    )

    assert high.theta > low.theta


def test_watchdog_is_adaptive_not_fixed_300() -> None:
    watchdog = AdaptiveWatchdog()

    small = watchdog.estimate(
        prompt_tokens=100,
        requested_output_tokens=40,
        observed_tokens_per_second=10,
    )

    large = watchdog.estimate(
        prompt_tokens=3000,
        requested_output_tokens=500,
        observed_tokens_per_second=10,
    )

    assert large > small
    assert small != 300
    assert large != 300


def test_existing_memory_projects_into_heartbeat() -> None:
    now = datetime.now(timezone.utc)

    memory = MemoryRecord(
        id=7,
        session_id="heartbeat-test",
        kind=MemoryKind.SEMANTIC,
        state=MemoryState.CONSOLIDATED,
        evidence_status=EvidenceStatus.VERIFIED,
        content="Hypercube heartbeat evidence",
        trace_hint="",
        source="test",
        destination="heartbeat",
        tags=("heartbeat",),
        salience=0.9,
        confidence=0.9,
        arousal=0.2,
        repetition_count=4,
        retrieval_count=3,
        protected=True,
        created_at=now,
        last_seen_at=now,
    )

    node = node_from_memory(
        memory,
        relevance=0.8,
    )

    assert node.key == "memory:7"
    assert node.evidence == 1.0
    assert node.protected
    assert node.weight > 0.5
