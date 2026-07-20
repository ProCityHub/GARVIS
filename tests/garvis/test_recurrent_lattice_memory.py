import pytest

from garvis.lattice_cognition import (
    MemoryAttractor,
    MirrorConnection,
    PulseBus,
    RecurrentLatticeMemory,
)


def build_memory() -> RecurrentLatticeMemory:
    return RecurrentLatticeMemory(
        nodes=(
            "negative-result",
            "evidence",
            "claim-discipline",
        ),
        mirrors=(
            MirrorConnection(
                left="negative-result",
                right="evidence",
                left_to_right=0.9,
                right_to_left=0.9,
            ),
            MirrorConnection(
                left="evidence",
                right="claim-discipline",
                left_to_right=0.8,
                right_to_left=0.8,
            ),
        ),
        attractors=(
            MemoryAttractor(
                attractor_id="learn-from-negative-result",
                pattern=(
                    ("negative-result", 1.0),
                    ("evidence", 0.7),
                    ("claim-discipline", 0.5),
                ),
                recall_threshold=0.9,
                provenance_refs=(
                    "hypercubeheartbeat:ultimatum-negative-result",
                ),
            ),
        ),
        persistence=0.4,
        inhibition=0.05,
        convergence_tolerance=0.001,
        max_cycles=32,
    )


def test_mirror_requires_both_directions() -> None:
    with pytest.raises(ValueError):
        MirrorConnection(
            left="a",
            right="b",
            left_to_right=1.0,
            right_to_left=0.0,
        )


def test_partial_cue_reconstructs_distributed_attractor() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)

    result = memory.recall(
        cue={"negative-result": 1.0},
        pulse=pulse,
    )

    state = result.state_dict()

    assert result.recalled is True
    assert result.recalled_attractor_id == (
        "learn-from-negative-result"
    )
    assert result.similarity >= 0.9
    assert state["negative-result"] > 0.0
    assert state["evidence"] > 0.0
    assert state["claim-discipline"] > 0.0
    assert result.external_action_allowed is False


def test_reciprocal_path_propagates_in_reverse() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)

    result = memory.recall(
        cue={"claim-discipline": 1.0},
        pulse=pulse,
    )

    state = result.state_dict()

    assert state["claim-discipline"] > 0.0
    assert state["evidence"] > 0.0
    assert state["negative-result"] > 0.0


def test_state_remains_bounded() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)

    result = memory.recall(
        cue={
            "negative-result": 1.0,
            "evidence": 1.0,
            "claim-discipline": 1.0,
        },
        pulse=pulse,
    )

    assert all(
        0.0 <= value <= 1.0
        for value in result.state_dict().values()
    )


def test_recall_is_deterministic() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)
    cue = {"negative-result": 1.0}

    first = memory.recall(cue=cue, pulse=pulse)
    second = memory.recall(cue=cue, pulse=pulse)

    assert first == second


def test_topology_hash_is_deterministic() -> None:
    first = build_memory()

    second = RecurrentLatticeMemory(
        nodes=tuple(reversed(first.nodes)),
        mirrors=tuple(reversed(first.mirrors)),
        attractors=first.attractors,
        persistence=first.persistence,
        inhibition=first.inhibition,
        convergence_tolerance=first.convergence_tolerance,
        max_cycles=first.max_cycles,
    )

    assert first.compute_sha256() == second.compute_sha256()


def test_unknown_cue_node_is_rejected() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)

    with pytest.raises(ValueError):
        memory.recall(
            cue={"unknown-node": 1.0},
            pulse=pulse,
        )


def test_empty_cue_is_rejected() -> None:
    memory = build_memory()
    pulse = PulseBus().emit(cycle=1)

    with pytest.raises(ValueError):
        memory.recall(
            cue={},
            pulse=pulse,
        )


def test_memory_contains_no_execution_interface() -> None:
    memory = build_memory()

    assert not hasattr(memory, "execute")
    assert not hasattr(memory, "send")
    assert not hasattr(memory, "connect")
