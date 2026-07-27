"""Canonical mathematics for Adrien D. Thomas's GARVIS Hypercube Heartbeat.

Creator / conceptual architect: Adrien D. Thomas (ProCityHub/GARVIS).

This module separates five mathematical layers that had previously been mixed:

1. semantic cube: eight 3-bit interpretation perspectives;
2. tesseract lift: the semantic cube plus a recurrence/return axis;
3. nine-phase Heartbeat: a circle R / 1.8Z with 0.2 phase spacing;
4. bounded recurrence: an evidence/state smoothing operator;
5. O/A/B semantic descriptor: Observer, Actor, Background with PHI exponents.

The semantic O/A/B descriptor is a framework quantity, not a truth,
intelligence, consciousness, or AGI score. It is intentionally separate from
boundary-bridge recurrence. Prime-lattice and quantum mappings remain
experimental research layers rather than universal physical laws.

Python 3.9 compatible; standard library only.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

CREATOR = "Adrien D. Thomas"
PROJECT = "ProCityHub/GARVIS"

PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI
INV_PHI_SQ = 1.0 / (PHI * PHI)
TAU = 2.0 * math.pi

# Heartbeat coordinates.
OBSERVER_ORIGIN = 0.0
COHERENCE_PLANE = 0.6
ENERGY_PLANE = 1.0
PHASE_STEP = 0.2
CYCLE_SPAN = 1.8
BOUNDARY_WIDTH = 0.4
HEARTBEAT_PHASE_COUNT = 9
PHASE_NAMES = (
    "RECEIVE",
    "SEGMENT",
    "PREDICT",
    "VERIFY",
    "SIMULATE",
    "PLAN",
    "OUTPUT",
    "FEEDBACK",
    "CONSOLIDATE",
)
PHASE_COORDINATES = tuple(index * PHASE_STEP for index in range(HEARTBEAT_PHASE_COUNT))
ANGULAR_STEP = TAU / HEARTBEAT_PHASE_COUNT

# Semantic cube perspectives. Bits are deliberately preserved as first-class
# structure so neighborhood, complement, and distance are exact operations.
PERSPECTIVES = {
    "000": "Literal",
    "001": "Context",
    "010": "Intent",
    "011": "Relation",
    "100": "Evidence",
    "101": "Possibility",
    "110": "Consequence",
    "111": "Integration",
}

# Prime-lattice topology used by the existing research program.
PRIME_LATTICE_CORNERS = 8
PRIME_LATTICE_WALLS = 6
PRIME_LATTICE_POLARITIES = 2
PRIME_LATTICE_EPOCH = PRIME_LATTICE_CORNERS * PRIME_LATTICE_WALLS * PRIME_LATTICE_POLARITIES

_EPSILON = 1e-12


class HypercubeMathError(ValueError):
    """Raised when a value violates the deterministic mathematical contract."""


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise HypercubeMathError("{} must be finite".format(name))
    return number


def clamp01(value: float) -> float:
    """Clamp a finite value into [0, 1]."""

    return max(0.0, min(1.0, _finite(value, "value")))


# ---------------------------------------------------------------------------
# Generic n-cube / tesseract structure
# ---------------------------------------------------------------------------


def n_cube_face_count(dimension: int, face_dimension: int) -> int:
    """Number of k-faces of an n-cube: 2^(n-k) * C(n,k)."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
        raise HypercubeMathError("dimension must be a non-negative integer")
    if (
        isinstance(face_dimension, bool)
        or not isinstance(face_dimension, int)
        or not 0 <= face_dimension <= dimension
    ):
        raise HypercubeMathError("face_dimension must be an integer in [0, dimension]")
    return (2 ** (dimension - face_dimension)) * math.comb(dimension, face_dimension)


def n_cube_f_vector(dimension: int) -> Tuple[int, ...]:
    """Return counts for vertices, edges, ..., full n-cell."""

    return tuple(n_cube_face_count(dimension, k) for k in range(dimension + 1))


def binary_vertices(dimension: int) -> Tuple[Tuple[int, ...], ...]:
    """All {0,1}^n vertices in lexicographic order."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
        raise HypercubeMathError("dimension must be a non-negative integer")
    return tuple(itertools.product((0, 1), repeat=dimension))


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Graph distance on an n-cube when both inputs are binary vertices."""

    if len(left) != len(right):
        raise HypercubeMathError("vertices must have equal dimension")
    for value in tuple(left) + tuple(right):
        if value not in (0, 1):
            raise HypercubeMathError("vertices must contain only 0 and 1")
    return sum(1 for a, b in zip(left, right) if a != b)


def are_adjacent(left: Sequence[int], right: Sequence[int]) -> bool:
    """Two n-cube vertices share an edge iff their Hamming distance is one."""

    return hamming_distance(left, right) == 1


def antipode(vertex: Sequence[int]) -> Tuple[int, ...]:
    """Exact n-cube antipode: bitwise complement."""

    if any(value not in (0, 1) for value in vertex):
        raise HypercubeMathError("vertex must contain only 0 and 1")
    return tuple(1 - value for value in vertex)


def n_cube_edges(dimension: int) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Enumerate each n-cube edge once."""

    vertices = binary_vertices(dimension)
    edges = []
    for index, left in enumerate(vertices):
        for right in vertices[index + 1 :]:
            if are_adjacent(left, right):
                edges.append((left, right))
    return tuple(edges)


def n_cube_geometry(dimension: int, side: float = 1.0) -> Dict[str, float]:
    """Basic Euclidean geometry for a centered n-cube of side length s."""

    length = _finite(side, "side")
    if length <= 0.0:
        raise HypercubeMathError("side must be positive")
    return {
        "dimension": float(dimension),
        "side": length,
        "hypervolume": length ** dimension,
        "circumradius": 0.5 * length * math.sqrt(dimension),
        "cell_hyperplane_distance": 0.5 * length,
        "graph_diameter": float(dimension),
        "degree": float(dimension),
    }


def tesseract_report(side: float = 1.0) -> Dict[str, object]:
    """Exact combinatorial and metric invariants of the 4-cube."""

    f_vector = n_cube_f_vector(4)
    geometry = n_cube_geometry(4, side)
    return {
        "dimension": 4,
        "vertices": f_vector[0],
        "edges": f_vector[1],
        "square_faces": f_vector[2],
        "cubic_cells": f_vector[3],
        "four_cells": f_vector[4],
        "f_vector": f_vector,
        "degree": 4,
        "diameter": 4,
        "circumradius": geometry["circumradius"],
        "hypervolume": geometry["hypervolume"],
    }


# ---------------------------------------------------------------------------
# Eight-perspective semantic cube and 4D recurrence lift
# ---------------------------------------------------------------------------


def perspective_bits(code: str) -> Tuple[int, int, int]:
    if code not in PERSPECTIVES:
        raise HypercubeMathError("unknown perspective code: {}".format(code))
    return tuple(int(char) for char in code)  # type: ignore[return-value]


def perspective_neighbors(code: str) -> Tuple[str, ...]:
    """Return the three semantic-cube neighbors of one perspective."""

    bits = perspective_bits(code)
    result = []
    for index in range(3):
        candidate = list(bits)
        candidate[index] ^= 1
        result.append("".join(str(value) for value in candidate))
    return tuple(sorted(result))


def perspective_complement(code: str) -> str:
    """Opposite corner in the 3-cube."""

    return "".join(str(value) for value in antipode(perspective_bits(code)))


def normalize_perspective_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Normalize a non-negative semantic field over the eight cube vertices."""

    values = {}
    total = 0.0
    for code in PERSPECTIVES:
        value = _finite(weights.get(code, 0.0), "perspective weight")
        if value < 0.0:
            raise HypercubeMathError("perspective weights must be non-negative")
        values[code] = value
        total += value
    extra = set(weights).difference(PERSPECTIVES)
    if extra:
        raise HypercubeMathError("unknown perspective weights: {}".format(", ".join(sorted(extra))))
    if total <= 0.0:
        raise HypercubeMathError("perspective field must contain positive mass")
    return {code: value / total for code, value in values.items()}


def semantic_barycenter(weights: Mapping[str, float]) -> Tuple[float, float, float]:
    """Weighted center of a semantic field in the unit 3-cube."""

    field = normalize_perspective_weights(weights)
    axes = [0.0, 0.0, 0.0]
    for code, mass in field.items():
        bits = perspective_bits(code)
        for axis, bit in enumerate(bits):
            axes[axis] += mass * bit
    return tuple(axes)  # type: ignore[return-value]


def semantic_entropy_bits(weights: Mapping[str, float]) -> float:
    """Shannon entropy of normalized semantic activation over eight perspectives."""

    field = normalize_perspective_weights(weights)
    return -sum(mass * math.log(mass, 2) for mass in field.values() if mass > 0.0)


def semantic_dirichlet_energy(weights: Mapping[str, float]) -> float:
    """Graph roughness: sum over cube edges of squared activation difference.

    Low energy means neighboring perspectives carry similar activation. High
    energy means the field is sharply separated across semantic relationships.
    This is descriptive graph mathematics, not evidence strength.
    """

    field = normalize_perspective_weights(weights)
    energy = 0.0
    seen = set()
    for code in PERSPECTIVES:
        for neighbor in perspective_neighbors(code):
            edge = tuple(sorted((code, neighbor)))
            if edge in seen:
                continue
            seen.add(edge)
            difference = field[code] - field[neighbor]
            energy += difference * difference
    return energy


def lift_perspective(code: str, return_side: int) -> Tuple[int, int, int, int]:
    """Lift one semantic perspective into the 4-cube recurrence axis.

    return_side=0 denotes the outward/current side; return_side=1 denotes the
    recurrence/return side. Toggling only this last bit is an OAB boundary pair.
    Complementing all four bits is the geometric tesseract antipode.
    """

    if return_side not in (0, 1):
        raise HypercubeMathError("return_side must be 0 or 1")
    return perspective_bits(code) + (return_side,)


def boundary_pair(vertex4: Sequence[int]) -> Tuple[int, int, int, int]:
    """Toggle only the recurrence axis of a lifted semantic vertex."""

    if len(vertex4) != 4 or any(value not in (0, 1) for value in vertex4):
        raise HypercubeMathError("vertex4 must be a four-bit vertex")
    return tuple(vertex4[:3]) + (1 - vertex4[3],)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Nine-phase Heartbeat and bounded recurrence
# ---------------------------------------------------------------------------


def phase_wrap(value: float) -> float:
    """Wrap any coordinate into [0, 1.8)."""

    wrapped = _finite(value, "phase") % CYCLE_SPAN
    if abs(wrapped) < _EPSILON or abs(wrapped - CYCLE_SPAN) < _EPSILON:
        return OBSERVER_ORIGIN
    return round(wrapped, 12)


def phase_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise HypercubeMathError("phase index must be an integer")
    return index % HEARTBEAT_PHASE_COUNT


def phase_coordinate(index: int) -> float:
    return PHASE_COORDINATES[phase_index(index)]


def heartbeat_angle(index: int) -> float:
    return phase_index(index) * ANGULAR_STEP


def mirror_phase_index(index: int) -> int:
    """Two-way Heartbeat mirror: 0<->8, 1<->7, 2<->6, 3<->5, 4<->4."""

    return HEARTBEAT_PHASE_COUNT - 1 - phase_index(index)


def oab_wrap_phase_index(index: int) -> int:
    """Advance one Heartbeat step, wrapping 8 -> 0."""

    return (phase_index(index) + 1) % HEARTBEAT_PHASE_COUNT


def reciprocal_phase_index(index: int) -> int:
    """Mirror then OAB-wrap: W(mu(k)) = -k mod 9."""

    return (-phase_index(index)) % HEARTBEAT_PHASE_COUNT


def circular_distance(left: float, right: float) -> float:
    left_w = phase_wrap(left)
    right_w = phase_wrap(right)
    direct = abs(left_w - right_w)
    return min(direct, CYCLE_SPAN - direct)


def boundary_overlap(phase_value: float) -> float:
    """Triangular CONSOLIDATE-to-RECEIVE overlap around observer origin."""

    distance = circular_distance(phase_value, OBSERVER_ORIGIN)
    return clamp01(1.0 - distance / BOUNDARY_WIDTH)


def observer_gain(phase_value: float) -> float:
    """Observer carry gain from coherence plane 0.6 toward energy plane 1.0."""

    return COHERENCE_PLANE + (ENERGY_PLANE - COHERENCE_PLANE) * boundary_overlap(phase_value)


def recurrence_weights(phase_value: float, actor_strength: float = 1.0) -> Tuple[float, float]:
    """Return normalized (previous, observation) weights for U."""

    strength = clamp01(actor_strength)
    gain = observer_gain(phase_value)
    if strength == 0.0:
        return 1.0, 0.0
    denominator = gain + strength
    return gain / denominator, strength / denominator


def transition(previous: float, observed: float, phase_value: float, actor_strength: float = 1.0) -> float:
    """Bounded recurrence U(x,y;p,a) for scalar metadata.

    U = [gO(p) x + a y] / [gO(p)+a].
    This is a recurrence/smoothing law. It is not a truth score.
    """

    old = clamp01(previous)
    new = clamp01(observed)
    old_weight, new_weight = recurrence_weights(phase_value, actor_strength)
    return clamp01(old_weight * old + new_weight * new)


def cycle_memory_retention(actor_strength: float = 1.0) -> float:
    """Product of previous-state coefficients across one nine-phase cycle.

    For repeated U updates with a fixed observation, this is the exact linear
    coefficient multiplying the starting deviation after one complete cycle.
    """

    coefficient = 1.0
    for coordinate in PHASE_COORDINATES:
        previous_weight, _ = recurrence_weights(coordinate, actor_strength)
        coefficient *= previous_weight
    return coefficient


# ---------------------------------------------------------------------------
# O/A/B semantic descriptor (Observer / Actor / Background)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticOAB:
    """Bounded semantic inputs for the Adrien-framework O/A/B relation.

    B means Background/environment here. It is intentionally not the boundary
    Bridge object used by the recurrence engine.
    """

    observer: float
    actor: float
    background: float

    def bounded(self) -> "SemanticOAB":
        return SemanticOAB(
            clamp01(self.observer),
            clamp01(self.actor),
            clamp01(self.background),
        )


def semantic_exponents() -> Tuple[float, float, float]:
    """Canonical elasticities: 1, 1/phi, 1/phi^2; their sum is exactly 2."""

    return 1.0, INV_PHI, INV_PHI_SQ


def semantic_coupling_raw(oab: SemanticOAB) -> float:
    """C = O^1 * A^(1/phi) * B^(1/phi^2), kept as framework metadata.

    Because all exponents are positive, bounded inputs yield C in [0,1].
    No empirical or decision meaning is attached to C by this function.
    """

    bounded = oab.bounded()
    return (
        bounded.observer
        * (bounded.actor ** INV_PHI)
        * (bounded.background ** INV_PHI_SQ)
    )


def semantic_coupling_normalized(oab: SemanticOAB) -> float:
    """Degree-normalized geometric form sqrt(C).

    Since 1 + 1/phi + 1/phi^2 = 2, sqrt(C) has exponents summing to one and is
    therefore a weighted geometric mean on positive inputs.
    """

    return math.sqrt(semantic_coupling_raw(oab))


def semantic_elasticities() -> Dict[str, float]:
    """Relative log sensitivities of the raw descriptor."""

    return {
        "observer": 1.0,
        "actor": INV_PHI,
        "background": INV_PHI_SQ,
    }


# ---------------------------------------------------------------------------
# Prime-lattice topology (research mapping, not a universal prime law)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimeLatticeAddress:
    ordinal: int
    corner: int
    wall: int
    polarity: int
    epoch: int


def prime_lattice_address(ordinal: int) -> PrimeLatticeAddress:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise HypercubeMathError("ordinal must be a non-negative integer")
    local = ordinal % PRIME_LATTICE_EPOCH
    return PrimeLatticeAddress(
        ordinal=ordinal,
        corner=local % PRIME_LATTICE_CORNERS,
        wall=(local // PRIME_LATTICE_CORNERS) % PRIME_LATTICE_WALLS,
        polarity=(local // (PRIME_LATTICE_CORNERS * PRIME_LATTICE_WALLS))
        % PRIME_LATTICE_POLARITIES,
        epoch=ordinal // PRIME_LATTICE_EPOCH,
    )


def prime_lattice_antipode_ordinal(ordinal: int) -> int:
    address = prime_lattice_address(ordinal)
    local = ordinal % PRIME_LATTICE_EPOCH
    return address.epoch * PRIME_LATTICE_EPOCH + (PRIME_LATTICE_EPOCH - 1 - local)


def prime_lattice_antipode(address: PrimeLatticeAddress) -> Tuple[int, int, int]:
    return (
        PRIME_LATTICE_CORNERS - 1 - address.corner,
        PRIME_LATTICE_WALLS - 1 - address.wall,
        PRIME_LATTICE_POLARITIES - 1 - address.polarity,
    )


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------


def invariant_report() -> Dict[str, object]:
    """Machine-readable exact structure used by GARVIS research code."""

    exponent_sum = sum(semantic_exponents())
    return {
        "creator": CREATOR,
        "project": PROJECT,
        "semantic_cube": {
            "vertices": len(PERSPECTIVES),
            "edges": n_cube_face_count(3, 1),
            "degree": 3,
            "perspectives": dict(PERSPECTIVES),
            "uniform_entropy_bits": 3.0,
            "field_statistics": ("barycenter", "entropy_bits", "dirichlet_energy"),
        },
        "tesseract": tesseract_report(),
        "heartbeat": {
            "phase_count": HEARTBEAT_PHASE_COUNT,
            "cycle_span": CYCLE_SPAN,
            "phase_step": PHASE_STEP,
            "coordinates": PHASE_COORDINATES,
            "angle_step": ANGULAR_STEP,
            "mirror_identity": "mu(k)=8-k",
            "oab_wrap_identity": "W(k)=k+1 mod 9",
            "reciprocal_identity": "W(mu(k))=-k mod 9",
            "boundary_width": BOUNDARY_WIDTH,
            "coherence_plane": COHERENCE_PLANE,
            "energy_plane": ENERGY_PLANE,
        },
        "semantic_oab": {
            "meaning": "Observer / Actor / Background",
            "exponents": semantic_exponents(),
            "exponent_sum": exponent_sum,
            "raw_descriptor": "O^1 * A^(1/phi) * B^(1/phi^2)",
            "normalized_descriptor": "sqrt(raw_descriptor)",
            "decision_score": False,
        },
        "prime_lattice": {
            "corners": PRIME_LATTICE_CORNERS,
            "walls": PRIME_LATTICE_WALLS,
            "polarities": PRIME_LATTICE_POLARITIES,
            "epoch_nodes": PRIME_LATTICE_EPOCH,
            "status": "experimental topology mapping",
        },
        "scientific_boundary": {
            "agi_established": False,
            "consciousness_established": False,
            "universal_physics_claim": False,
            "repetition_is_evidence": False,
        },
    }


__all__ = [
    "ANGULAR_STEP",
    "BOUNDARY_WIDTH",
    "COHERENCE_PLANE",
    "CYCLE_SPAN",
    "ENERGY_PLANE",
    "HEARTBEAT_PHASE_COUNT",
    "INV_PHI",
    "INV_PHI_SQ",
    "OBSERVER_ORIGIN",
    "PERSPECTIVES",
    "PHASE_COORDINATES",
    "PHASE_NAMES",
    "PHASE_STEP",
    "PHI",
    "PRIME_LATTICE_EPOCH",
    "PrimeLatticeAddress",
    "SemanticOAB",
    "antipode",
    "are_adjacent",
    "boundary_overlap",
    "boundary_pair",
    "binary_vertices",
    "circular_distance",
    "clamp01",
    "cycle_memory_retention",
    "hamming_distance",
    "heartbeat_angle",
    "invariant_report",
    "lift_perspective",
    "mirror_phase_index",
    "n_cube_edges",
    "n_cube_f_vector",
    "n_cube_face_count",
    "n_cube_geometry",
    "oab_wrap_phase_index",
    "observer_gain",
    "perspective_complement",
    "perspective_neighbors",
    "normalize_perspective_weights",
    "semantic_barycenter",
    "semantic_entropy_bits",
    "semantic_dirichlet_energy",
    "phase_coordinate",
    "phase_index",
    "phase_wrap",
    "prime_lattice_address",
    "prime_lattice_antipode",
    "prime_lattice_antipode_ordinal",
    "reciprocal_phase_index",
    "recurrence_weights",
    "semantic_coupling_normalized",
    "semantic_coupling_raw",
    "semantic_elasticities",
    "semantic_exponents",
    "tesseract_report",
    "transition",
]
