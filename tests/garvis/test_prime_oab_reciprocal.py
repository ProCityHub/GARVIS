from math import isclose, pi

import pytest

from garvis.prime_oab_reciprocal import (
    EPOCH_NODES,
    OABCapsule,
    PHASE_NAMES,
    RoleVector,
    epoch_addresses,
    first_epoch_prime_span,
    first_epoch_primes,
    heartbeat_math_claims,
    lattice_antipode,
    lattice_antipode_ordinal,
    mirror_phase_index,
    oab_wrap_phase_index,
    prime_gaps,
    prime_mirror_defects,
    reciprocal_phase_index,
)


def test_heartbeat_mirror_is_involution():
    for index in range(9):
        assert mirror_phase_index(mirror_phase_index(index)) == index


def test_oab_after_mirror_is_reciprocal_phase_order():
    for index in range(9):
        assert oab_wrap_phase_index(mirror_phase_index(index)) == reciprocal_phase_index(index)


def test_phase_pairs_match_two_way_mirror():
    pairs = {
        "RECEIVE": "CONSOLIDATE",
        "SEGMENT": "FEEDBACK",
        "PREDICT": "OUTPUT",
        "VERIFY": "PLAN",
        "SIMULATE": "SIMULATE",
    }
    for left, right in pairs.items():
        index = PHASE_NAMES.index(left)
        assert PHASE_NAMES[mirror_phase_index(index)] == right


def test_prime_lattice_epoch_has_96_unique_addresses():
    addresses = epoch_addresses()
    assert len(addresses) == EPOCH_NODES == 96
    topology = {(item.corner, item.wall, item.polarity) for item in addresses}
    assert len(topology) == 96
    assert addresses[0].prime == 2
    assert addresses[-1].prime == 503


def test_lattice_antipode_complements_topology():
    addresses = epoch_addresses()
    for item in addresses:
        mirror = addresses[lattice_antipode_ordinal(item.ordinal)]
        assert (mirror.corner, mirror.wall, mirror.polarity) == lattice_antipode(item)
        assert mirror.corner == 7 - item.corner
        assert mirror.wall == 5 - item.wall
        assert mirror.polarity == 1 - item.polarity


def test_first_epoch_prime_gaps_close_span():
    primes = first_epoch_primes()
    gaps = prime_gaps(primes)
    assert len(primes) == 96
    assert len(gaps) == 95
    assert sum(gaps) == first_epoch_prime_span() == 501


def test_prime_mirror_defect_field_is_exact_finite_window_property():
    defects = prime_mirror_defects()
    assert len(defects) == 48
    assert min(defects) == 0
    assert max(defects) == 73
    assert sum(defects) == 2201


def test_oab_capsule_reflects_roles_without_scalar_score():
    outward = RoleVector(tuple(index / 8.0 for index in range(9)))
    capsule = OABCapsule(outward)
    assert capsule.return_seed().values == tuple(reversed(outward.values))
    assert capsule.compare_return(capsule.return_seed()) == (0.0,) * 9


def test_machine_checkable_claims_are_plain_arithmetic():
    claims = heartbeat_math_claims()
    assert len(claims) >= 4
    for claim in claims:
        assert set(claim) == {"claim_id", "expression", "expected", "tolerance", "meaning"}
    by_id = {claim["claim_id"]: claim for claim in claims}
    assert by_id["prime_lattice_epoch_nodes"]["expression"] == "8*6*2"
    assert by_id["first_epoch_prime_span"]["expression"] == "503-2"


def test_retracted_scalar_phi_rule_is_not_executable():
    import ast
    import inspect
    import garvis.prime_oab_reciprocal as module

    tree = ast.parse(inspect.getsource(module))
    executable_names = {
        node.id.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "phi" not in executable_names
