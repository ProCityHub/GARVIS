"""Prime-OAB Reciprocal Hypercube research primitives.

Creator / conceptual architect: Adrien D. Thomas (ProCityHub/GARVIS).

This module encodes the deterministic mathematics used by the GARVIS
Hypercube Heartbeat research program:

- nine-phase Heartbeat coordinates;
- OAB wrap and reciprocal phase ordering;
- 96-address Prime Lattice topology;
- lattice antipodes and the finite Prime Mirror Defect field;
- typed outward/OAB/return role vectors.

Scientific boundary
-------------------
This is an experimental architecture module. It does not claim AGI,
consciousness, singularity, or a universal physical law.

The organization-wide scalar-PHI multiplier retraction is enforced here: no
scalar Lattice expression is used as a truth, intelligence,
consciousness, or decision score.

Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable, Sequence, Tuple


CYCLE_SPAN = 1.8
PHASE_STEP = 0.2
HEARTBEAT_PHASE_COUNT = 9
EPOCH_NODES = 96
CORNERS = 8
WALLS = 6
POLARITIES = 2

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
ANGULAR_STEP = 2.0 * pi / HEARTBEAT_PHASE_COUNT


class PrimeOABError(ValueError):
    """Raised when a Prime-OAB research value violates the deterministic contract."""


@dataclass(frozen=True)
class PhasePoint:
    """One canonical Heartbeat phase."""

    index: int
    name: str
    coordinate: float

    @property
    def mirror_index(self) -> int:
        return mirror_phase_index(self.index)

    @property
    def mirror_name(self) -> str:
        return PHASE_NAMES[self.mirror_index]

    @property
    def reciprocal_index(self) -> int:
        return reciprocal_phase_index(self.index)

    @property
    def reciprocal_name(self) -> str:
        return PHASE_NAMES[self.reciprocal_index]


@dataclass(frozen=True)
class PrimeLatticeAddress:
    """Prime identity plus ordinal-controlled topology."""

    prime: int
    ordinal: int
    corner: int
    wall: int
    polarity: int
    epoch: int

    @property
    def sign(self) -> str:
        return "+" if self.polarity == 0 else "-"

    @property
    def label(self) -> str:
        return "E{}:S{}:W{}:C{}".format(
            self.epoch,
            self.sign,
            self.wall,
            self.corner,
        )


@dataclass(frozen=True)
class RoleVector:
    """Nine bounded role values in Heartbeat order.

    The values are research metadata, not an AGI or consciousness score.
    """

    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != HEARTBEAT_PHASE_COUNT:
            raise PrimeOABError("role vector must contain exactly nine values")
        for value in self.values:
            if not 0.0 <= float(value) <= 1.0:
                raise PrimeOABError("role values must be bounded in [0, 1]")

    def mirrored(self) -> "RoleVector":
        """Reflect RECEIVE<->CONSOLIDATE, SEGMENT<->FEEDBACK, etc."""

        return RoleVector(tuple(reversed(self.values)))


@dataclass(frozen=True)
class OABCapsule:
    """Coherent/classical research representation of outward state retained at OAB."""

    outward: RoleVector

    def return_seed(self) -> RoleVector:
        """Seed the reciprocal return roles from complementary outward roles."""

        return self.outward.mirrored()

    def compare_return(self, returned: RoleVector) -> Tuple[float, ...]:
        """Absolute per-role residual against the mirror-seeded return.

        This is a diagnostic vector and is intentionally not collapsed into one
        truth/intelligence/consciousness number.
        """

        expected = self.return_seed().values
        return tuple(abs(left - right) for left, right in zip(expected, returned.values))


def _require_phase_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise PrimeOABError("phase index must be an integer")
    if not 0 <= index < HEARTBEAT_PHASE_COUNT:
        raise PrimeOABError("phase index must be between 0 and 8")
    return index


def phase_point(index: int) -> PhasePoint:
    index = _require_phase_index(index)
    return PhasePoint(index=index, name=PHASE_NAMES[index], coordinate=PHASE_COORDINATES[index])


def mirror_phase_index(index: int) -> int:
    """Heartbeat two-way mirror: 0<->8, 1<->7, 2<->6, 3<->5, 4<->4."""

    index = _require_phase_index(index)
    return HEARTBEAT_PHASE_COUNT - 1 - index


def oab_wrap_phase_index(index: int) -> int:
    """Advance one 0.2 phase with wrap from CONSOLIDATE back to RECEIVE."""

    index = _require_phase_index(index)
    return (index + 1) % HEARTBEAT_PHASE_COUNT


def reciprocal_phase_index(index: int) -> int:
    """OAB after reflection: W(mu(index)) == -index mod 9."""

    index = _require_phase_index(index)
    return (-index) % HEARTBEAT_PHASE_COUNT


def phase_coordinate(index: int) -> float:
    return phase_point(index).coordinate


def mirror_coordinate(index: int) -> float:
    return phase_coordinate(mirror_phase_index(index))


def wrapped_coordinate(index: int) -> float:
    return phase_coordinate(oab_wrap_phase_index(index))


def reciprocal_coordinate(index: int) -> float:
    return phase_coordinate(reciprocal_phase_index(index))


def first_primes(count: int) -> Tuple[int, ...]:
    """Return the first count prime numbers using deterministic trial division."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise PrimeOABError("prime count must be a non-negative integer")

    values = []
    candidate = 2
    while len(values) < count:
        limit = int(candidate ** 0.5)
        is_prime = True
        for prime in values:
            if prime > limit:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            values.append(candidate)
        candidate = 3 if candidate == 2 else candidate + 2
    return tuple(values)


def prime_lattice_address(ordinal: int, primes: Sequence[int] = ()) -> PrimeLatticeAddress:
    """Map zero-based prime ordinal to one 8x6x2 lattice address."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise PrimeOABError("prime ordinal must be a non-negative integer")

    if primes:
        if ordinal >= len(primes):
            raise PrimeOABError("provided prime sequence does not contain ordinal")
        prime = int(primes[ordinal])
    else:
        prime = first_primes(ordinal + 1)[ordinal]

    return PrimeLatticeAddress(
        prime=prime,
        ordinal=ordinal,
        corner=ordinal % CORNERS,
        wall=(ordinal // CORNERS) % WALLS,
        polarity=(ordinal // (CORNERS * WALLS)) % POLARITIES,
        epoch=ordinal // EPOCH_NODES,
    )


def epoch_addresses(epoch: int = 0) -> Tuple[PrimeLatticeAddress, ...]:
    """Return a complete 96-address Prime Lattice epoch."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise PrimeOABError("epoch must be a non-negative integer")

    start = epoch * EPOCH_NODES
    primes = first_primes(start + EPOCH_NODES)
    return tuple(
        prime_lattice_address(ordinal, primes)
        for ordinal in range(start, start + EPOCH_NODES)
    )


def lattice_antipode_ordinal(ordinal: int) -> int:
    """Reflect one ordinal through the centre of its 96-node epoch."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise PrimeOABError("prime ordinal must be a non-negative integer")
    epoch = ordinal // EPOCH_NODES
    local = ordinal % EPOCH_NODES
    return epoch * EPOCH_NODES + (EPOCH_NODES - 1 - local)


def lattice_antipode(address: PrimeLatticeAddress) -> Tuple[int, int, int]:
    """Return the complementary (corner, wall, polarity) within one epoch."""

    return (
        CORNERS - 1 - address.corner,
        WALLS - 1 - address.wall,
        POLARITIES - 1 - address.polarity,
    )


def first_epoch_primes() -> Tuple[int, ...]:
    """The 96 prime identities used by the canonical first lattice epoch."""

    return first_primes(EPOCH_NODES)


def prime_gaps(primes: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(int(value) for value in primes)
    if len(values) < 2:
        return ()
    return tuple(right - left for left, right in zip(values, values[1:]))


def first_epoch_prime_span() -> int:
    values = first_epoch_primes()
    return values[-1] - values[0]


def prime_gap_phase(gap: int, span: int = 0) -> float:
    """Map a prime gap into the canonical first-epoch 2*pi phase closure."""

    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
        raise PrimeOABError("prime gap must be a non-negative integer")
    denominator = span or first_epoch_prime_span()
    if denominator <= 0:
        raise PrimeOABError("phase span must be positive")
    return 2.0 * pi * gap / denominator


def prime_coordinate(prime: int) -> float:
    """Normalize a first-epoch prime identity from 2..503 into 0..1."""

    values = first_epoch_primes()
    if prime < values[0] or prime > values[-1]:
        raise PrimeOABError("prime lies outside the first 96-prime epoch")
    return (prime - values[0]) / float(values[-1] - values[0])


def prime_mirror_defects() -> Tuple[int, ...]:
    """Finite asymmetry field for the 48 antipodal pairs in the first epoch.

    Perfect value reflection would satisfy p_n + p_(95-n) == 2 + 503 == 505.
    D_n is the shortfall from that finite-window perfect reflection.

    This is an exact property of this chosen prime window, not a universal law.
    """

    values = first_epoch_primes()
    perfect_sum = values[0] + values[-1]
    return tuple(
        perfect_sum - (values[index] + values[EPOCH_NODES - 1 - index])
        for index in range(EPOCH_NODES // 2)
    )


def prime_mirror_defect_phase(defect: int) -> float:
    """Map one finite Prime Mirror Defect into the first-epoch phase span."""

    if isinstance(defect, bool) or not isinstance(defect, int) or defect < 0:
        raise PrimeOABError("mirror defect must be a non-negative integer")
    return 2.0 * pi * defect / first_epoch_prime_span()


def heartbeat_math_claims() -> Tuple[dict, ...]:
    """Machine-checkable arithmetic claims accepted by Research Hypercube Bridge.

    Symbolic claims are intentionally excluded because the repository verifier
    handles explicit arithmetic, not arbitrary theorem proving.
    """

    return (
        {
            "claim_id": "heartbeat_phase_count",
            "expression": "9",
            "expected": "9",
            "tolerance": "0",
            "meaning": "The Heartbeat contains nine canonical phases.",
        },
        {
            "claim_id": "heartbeat_coordinate_closure",
            "expression": "16/10 + 2/10",
            "expected": "1.8",
            "tolerance": "0",
            "meaning": "CONSOLIDATE plus one phase step reaches the 1.8 wrap boundary.",
        },
        {
            "claim_id": "prime_lattice_epoch_nodes",
            "expression": "8*6*2",
            "expected": "96",
            "tolerance": "0",
            "meaning": "One Prime Lattice epoch contains 96 unique topological addresses.",
        },
        {
            "claim_id": "first_epoch_prime_span",
            "expression": "503-2",
            "expected": "501",
            "tolerance": "0",
            "meaning": "The first 96-prime identity window spans 501 integer units.",
        },
    )


__all__ = [
    "ANGULAR_STEP",
    "CORNERS",
    "CYCLE_SPAN",
    "EPOCH_NODES",
    "HEARTBEAT_PHASE_COUNT",
    "OABCapsule",
    "PHASE_COORDINATES",
    "PHASE_NAMES",
    "PHASE_STEP",
    "POLARITIES",
    "PhasePoint",
    "PrimeLatticeAddress",
    "PrimeOABError",
    "RoleVector",
    "WALLS",
    "epoch_addresses",
    "first_epoch_prime_span",
    "first_epoch_primes",
    "first_primes",
    "heartbeat_math_claims",
    "lattice_antipode",
    "lattice_antipode_ordinal",
    "mirror_coordinate",
    "mirror_phase_index",
    "oab_wrap_phase_index",
    "phase_coordinate",
    "phase_point",
    "prime_coordinate",
    "prime_gap_phase",
    "prime_gaps",
    "prime_lattice_address",
    "prime_mirror_defect_phase",
    "prime_mirror_defects",
    "reciprocal_coordinate",
    "reciprocal_phase_index",
    "wrapped_coordinate",
]
