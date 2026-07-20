import pytest

from garvis.evidence_envelope import build_evidence_envelope
from garvis.psychology import (
    EvidenceSignalKind,
    adapt_evidence_envelope,
    evaluate_equilibrium,
)


def test_observed_evidence_receives_full_routing_weight() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={
            "result": {
                "auc_phi": 0.871258,
                "auc_flat": 0.861944,
                "delta": 0.009314,
            }
        },
        source_provenance_hashes={
            "result": "recorded-result-hash",
        },
    )

    assessment = adapt_evidence_envelope(envelope)

    assert len(assessment.signals) == 1
    assert assessment.observed_count == 1
    assert assessment.evidence_sufficiency == pytest.approx(1.0)
    assert assessment.evidence_error == pytest.approx(0.0)
    assert assessment.external_action_allowed is False

    signal = assessment.signals[0]

    assert signal.kind is EvidenceSignalKind.OBSERVED
    assert signal.routing_weight == pytest.approx(1.0)
    assert signal.provenance_hash == "recorded-result-hash"
    assert signal.external_action_allowed is False


def test_adapter_separates_inference_symbolism_and_speculation() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={"measured": 1},
        engineering_inference={"interpretation": "possible mechanism"},
        symbolic_interpretation={"symbol": "0.0 center"},
        unsupported_speculation={"idea": "unverified extension"},
        unknowns={"missing": "independent replication"},
    )

    assessment = adapt_evidence_envelope(envelope)

    assert assessment.observed_count == 1
    assert assessment.inferred_count == 1
    assert assessment.symbolic_count == 1
    assert assessment.speculative_count == 1
    assert assessment.unknown_count == 1

    expected = (1.0 + 0.5 + 0.25 + 0.1 + 0.0) / 5

    assert assessment.evidence_sufficiency == pytest.approx(expected)
    assert assessment.evidence_error == pytest.approx(1.0 - expected)
    assert assessment.unknown_fraction == pytest.approx(0.2)
    assert assessment.speculative_fraction == pytest.approx(0.2)


def test_adapter_output_is_deterministic() -> None:
    first = build_evidence_envelope(
        immutable_source_evidence={
            "b": {"value": 2},
            "a": {"value": 1},
        }
    )

    second = build_evidence_envelope(
        immutable_source_evidence={
            "a": {"value": 1},
            "b": {"value": 2},
        }
    )

    first_assessment = adapt_evidence_envelope(first)
    second_assessment = adapt_evidence_envelope(second)

    assert first_assessment == second_assessment
    assert [
        signal.signal_id
        for signal in first_assessment.signals
    ] == [
        signal.signal_id
        for signal in second_assessment.signals
    ]


def test_empty_envelope_fails_closed() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={}
    )

    assessment = adapt_evidence_envelope(envelope)

    assert assessment.signals == ()
    assert assessment.evidence_sufficiency == pytest.approx(0.0)
    assert assessment.evidence_error == pytest.approx(1.0)
    assert assessment.unknown_fraction == pytest.approx(1.0)
    assert assessment.external_action_allowed is False


def test_uncertain_evidence_reduces_equilibrium() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={},
        unsupported_speculation={"proposal": "unverified"},
        unknowns={"replication": "missing"},
    )

    assessment = adapt_evidence_envelope(envelope)

    result = evaluate_equilibrium(
        activation=1.0,
        wall_coherence=0.6,
        evidence=assessment.evidence_sufficiency,
        action_readiness=1.0,
        context_stability=1.0,
        tensions=assessment.equilibrium_tensions(),
        constraints_passed=True,
        external_action=False,
    )

    assert result.integrated_equilibrium < 0.95
    assert result.equilibrium_reached is False
    assert result.proposal_eligible is False


def test_adapter_does_not_mutate_the_source_envelope() -> None:
    envelope = build_evidence_envelope(
        immutable_source_evidence={
            "result": {"verdict": "NOT_SUPPORTED"}
        }
    )

    before = dict(envelope.immutable_source_evidence)

    adapt_evidence_envelope(envelope)

    after = dict(envelope.immutable_source_evidence)

    assert before == after
