import math

from garvis.hypercube_heartbeat_math import (
    PHI,
    PERSPECTIVES,
    SemanticOAB,
    antipode,
    boundary_overlap,
    boundary_pair,
    cycle_memory_retention,
    hamming_distance,
    lift_perspective,
    mirror_phase_index,
    n_cube_edges,
    n_cube_f_vector,
    oab_wrap_phase_index,
    perspective_complement,
    perspective_neighbors,
    semantic_barycenter,
    semantic_dirichlet_energy,
    semantic_entropy_bits,
    prime_lattice_address,
    prime_lattice_antipode,
    prime_lattice_antipode_ordinal,
    reciprocal_phase_index,
    semantic_coupling_normalized,
    semantic_coupling_raw,
    semantic_exponents,
    tesseract_report,
    transition,
)


def test_tesseract_invariants():
    assert n_cube_f_vector(4) == (16, 32, 24, 8, 1)
    report = tesseract_report(side=2.0)
    assert report["edges"] == 32
    assert report["circumradius"] == 2.0
    assert report["hypervolume"] == 16.0
    assert len(n_cube_edges(4)) == 32


def test_antipode_and_distance():
    vertex = (0, 1, 0, 1)
    opposite = antipode(vertex)
    assert opposite == (1, 0, 1, 0)
    assert hamming_distance(vertex, opposite) == 4


def test_semantic_cube_degree_and_complement():
    assert len(PERSPECTIVES) == 8
    for code in PERSPECTIVES:
        assert len(perspective_neighbors(code)) == 3
        assert perspective_complement(perspective_complement(code)) == code


def test_recurrence_axis_is_not_full_antipode():
    vertex = lift_perspective("100", 0)
    assert boundary_pair(vertex) == (1, 0, 0, 1)
    assert antipode(vertex) == (0, 1, 1, 1)


def test_phase_mirror_wrap_reciprocal_identity():
    for k in range(9):
        assert mirror_phase_index(k) == 8 - k
        assert oab_wrap_phase_index(k) == (k + 1) % 9
        assert reciprocal_phase_index(k) == (-k) % 9
        assert oab_wrap_phase_index(mirror_phase_index(k)) == reciprocal_phase_index(k)


def test_boundary_overlap_anchor_values():
    assert boundary_overlap(0.0) == 1.0
    assert math.isclose(boundary_overlap(0.2), 0.5)
    assert math.isclose(boundary_overlap(1.6), 0.5)
    assert boundary_overlap(0.4) == 0.0


def test_transition_is_bounded_and_idempotent():
    for phase in (0.0, 0.6, 1.6):
        value = transition(0.2, 0.8, phase, 1.0)
        assert 0.2 <= value <= 0.8
        assert transition(0.4, 0.4, phase, 1.0) == 0.4
    assert transition(0.2, 0.8, 0.0, 0.0) == 0.2


def test_full_cycle_retention_is_contractive_with_actor_input():
    retention = cycle_memory_retention(1.0)
    assert 0.0 < retention < 1.0
    assert cycle_memory_retention(0.0) == 1.0


def test_semantic_oab_exponents_and_normalized_descriptor():
    exponents = semantic_exponents()
    assert math.isclose(sum(exponents), 2.0, abs_tol=1e-12)
    assert math.isclose(exponents[1], 1.0 / PHI, abs_tol=1e-12)
    oab = SemanticOAB(0.81, 0.64, 0.49)
    raw = semantic_coupling_raw(oab)
    normalized = semantic_coupling_normalized(oab)
    assert 0.0 <= raw <= 1.0
    assert math.isclose(normalized * normalized, raw, rel_tol=1e-12)


def test_prime_lattice_epoch_and_antipode():
    addresses = {(
        prime_lattice_address(n).corner,
        prime_lattice_address(n).wall,
        prime_lattice_address(n).polarity,
    ) for n in range(96)}
    assert len(addresses) == 96
    for n in range(96):
        address = prime_lattice_address(n)
        other = prime_lattice_address(prime_lattice_antipode_ordinal(n))
        assert prime_lattice_antipode(address) == (other.corner, other.wall, other.polarity)


def test_semantic_field_graph_statistics():
    uniform = {code: 1.0 for code in PERSPECTIVES}
    assert semantic_barycenter(uniform) == (0.5, 0.5, 0.5)
    assert math.isclose(semantic_entropy_bits(uniform), 3.0)
    assert semantic_dirichlet_energy(uniform) == 0.0

    literal = {"000": 1.0}
    assert semantic_barycenter(literal) == (0.0, 0.0, 0.0)
    assert semantic_entropy_bits(literal) == 0.0
    assert semantic_dirichlet_energy(literal) > 0.0


def test_v13_reciprocal_consistency_if_present():
    try:
        from garvis import prime_oab_reciprocal as v13
    except ImportError:
        return
    for k in range(9):
        assert mirror_phase_index(k) == v13.mirror_phase_index(k)
        assert oab_wrap_phase_index(k) == v13.oab_wrap_phase_index(k)
        assert reciprocal_phase_index(k) == v13.reciprocal_phase_index(k)
    for n in (0, 1, 47, 48, 95, 96, 191):
        local = prime_lattice_address(n)
        old = v13.prime_lattice_address(n)
        assert (local.corner, local.wall, local.polarity, local.epoch) == (
            old.corner, old.wall, old.polarity, old.epoch
        )
