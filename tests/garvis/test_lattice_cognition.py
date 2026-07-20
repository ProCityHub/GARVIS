import pytest

from garvis.lattice_cognition import (
    LatticeResonancePacket,
    PermissionClass,
    PulseBus,
    PulsePhase,
    ResonanceComponent,
)


def build_packet() -> LatticeResonancePacket:
    return LatticeResonancePacket(
        packet_id="ultimatum-memory",
        cycle=0,
        components=(
            ResonanceComponent(
                concept_id="negative-result",
                amplitude=3.0,
                associations=("evidence", "learning"),
            ),
            ResonanceComponent(
                concept_id="phi-model",
                amplitude=1.0,
                phase_position=0.25,
                associations=("hypothesis",),
            ),
        ),
        energy=0.4,
        confidence=0.9,
        uncertainty=0.1,
        decay_rate=0.05,
        provenance_refs=("hypercubeheartbeat:ultimatum",),
        permission_class=PermissionClass.LOCAL_REASONING,
    )


def test_packet_normalizes_classical_components() -> None:
    packet = build_packet()

    assert sum(
        component.amplitude
        for component in packet.normalized_components
    ) == pytest.approx(1.0)

    assert packet.dominant_component.concept_id == "negative-result"
    assert packet.effective_component_count == pytest.approx(1.6)


def test_packet_hash_is_independent_of_component_order() -> None:
    first = build_packet()

    second = LatticeResonancePacket(
        packet_id=first.packet_id,
        cycle=first.cycle,
        components=tuple(reversed(first.components)),
        energy=first.energy,
        confidence=first.confidence,
        uncertainty=first.uncertainty,
        decay_rate=first.decay_rate,
        provenance_refs=first.provenance_refs,
        permission_class=first.permission_class,
    )

    assert first.compute_sha256() == second.compute_sha256()


def test_canonical_pulse_preserves_and_normalizes_1_point_6() -> None:
    pulse = PulseBus().emit(
        cycle=1,
        activation=1.0,
        wall_coherence=0.6,
    )

    assert pulse.raw_union == pytest.approx(1.6)
    assert pulse.normalized_center == pytest.approx(1.0)
    assert pulse.phase is PulsePhase.ACTIVATE


def test_partial_cue_recalls_matching_packet() -> None:
    packet = build_packet()
    bus = PulseBus()
    pulse = bus.emit(cycle=1)

    response = bus.transmit(
        packet=packet,
        pulse=pulse,
        cue_weights={
            "negative-result": 1.0,
            "phi-model": 0.2,
        },
        reflection=0.6,
        inhibition=0.0,
        recall_threshold=0.5,
    )

    assert response.cue_alignment == pytest.approx(0.8)
    assert response.energy_after_inhibition <= 1.0
    assert response.recall_strength >= 0.5
    assert response.recalled is True
    assert response.external_action_allowed is False
    assert response.updated_packet.cycle == 1


def test_full_inhibition_blocks_recall() -> None:
    packet = build_packet()
    bus = PulseBus()

    response = bus.transmit(
        packet=packet,
        pulse=bus.emit(cycle=1),
        cue_weights={"negative-result": 1.0},
        inhibition=1.0,
        recall_threshold=0.1,
    )

    assert response.energy_after_inhibition == pytest.approx(0.0)
    assert response.recall_strength == pytest.approx(0.0)
    assert response.recalled is False


def test_decay_reduces_energy_without_deleting_memory_structure() -> None:
    packet = build_packet()
    decayed = packet.decayed(cycles=3)

    assert decayed.energy < packet.energy
    assert decayed.components == packet.components
    assert decayed.provenance_refs == packet.provenance_refs
    assert decayed.cycle == 3


def test_packet_has_no_execution_permission() -> None:
    assert not hasattr(PermissionClass, "EXECUTE")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy", -0.1),
        ("energy", 1.1),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("uncertainty", -0.1),
        ("uncertainty", 1.1),
        ("decay_rate", -0.1),
        ("decay_rate", 1.1),
    ],
)
def test_invalid_unit_values_are_rejected(
    field: str,
    value: float,
) -> None:
    values = {
        "packet_id": "packet",
        "cycle": 0,
        "components": (
            ResonanceComponent(
                concept_id="concept",
                amplitude=1.0,
            ),
        ),
        "energy": 0.5,
        "confidence": 0.5,
        "uncertainty": 0.5,
        "decay_rate": 0.5,
    }

    values[field] = value

    with pytest.raises(ValueError):
        LatticeResonancePacket(**values)
