"""Lattice cycle scores.

Canonical ``C`` is ``hypercube_brain.core.coherence``. This module imports
that function and does not redefine it:

    C = O^1 * A^(1/phi) * B^(1/phi^2)

The half-exponent control is reported beside it and is not a replacement:

    C_unif = O * A^(1/2) * B^(1/2)

Both are a heuristic (``HYPOTHESIS_UNDER_TEST``). Neither overrides Law 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from hypercube_brain.core import coherence as coherence

FORMULA = "C = O^1 * A^(1/phi) * B^(1/phi^2)"
CONTROL_FORMULA = "C_unif = O * A^(1/2) * B^(1/2)"
HYPOTHESIS = "HYPOTHESIS_UNDER_TEST"

_STATE_BY_BITS = {
    "000": "DORMANT",
    "001": "PARTIAL",
    "010": "PARTIAL",
    "011": "PARTIAL",
    "100": "PARTIAL",
    "101": "PARTIAL",
    "110": "FRAGMENTED",
    "111": "INTEGRATED",
}


def _finite01(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def c_unif(observer: float, actor: float, background: float) -> float:
    """Half-exponent control. Inputs are clamped to [0, 1]."""

    observed = _finite01(observer)
    acted = _finite01(actor)
    linked = _finite01(background)
    score = observed * (acted ** 0.5) * (linked ** 0.5)
    return _finite01(score)


@dataclass(frozen=True)
class ScoreReport:
    o: float
    a: float
    b: float
    c: float
    c_unif: float
    stabilized_c: float
    bit_code: str
    bit_label: str
    state: str
    integrated: bool
    any_axis_below_0_3: bool
    phi_changes_decision: bool
    formula: str = FORMULA
    control_formula: str = CONTROL_FORMULA
    hypothesis: str = HYPOTHESIS

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _bits(observed: float, acted: float, linked: float) -> str:
    return (
        ("1" if observed >= 0.5 else "0")
        + ("1" if acted >= 0.5 else "0")
        + ("1" if linked >= 0.5 else "0")
    )


def score_axes(
    observer: float,
    actor: float,
    background: float,
    *,
    law_zero_ok: bool = True,
) -> ScoreReport:
    """Score one batch and name the state Law 0 still permits."""

    observed = _finite01(observer)
    acted = _finite01(actor)
    linked = _finite01(background)
    canonical = coherence(observed, acted, linked)
    control = c_unif(observed, acted, linked)
    bit_code = _bits(observed, acted, linked)
    bit_label = _STATE_BY_BITS[bit_code]
    below = observed < 0.3 or acted < 0.3 or linked < 0.3
    scalar_split = (canonical >= 0.5) != (control >= 0.5)

    if not law_zero_ok:
        state = "NOT_INTEGRATED"
        stabilized = 0.0
        integrated = False
    elif below:
        state = "FRAGMENTED"
        stabilized = 0.0
        integrated = False
    else:
        state = bit_label
        stabilized = canonical
        integrated = state == "INTEGRATED"

    return ScoreReport(
        o=observed,
        a=acted,
        b=linked,
        c=canonical,
        c_unif=control,
        stabilized_c=stabilized,
        bit_code=bit_code,
        bit_label=bit_label,
        state=state,
        integrated=integrated,
        any_axis_below_0_3=below,
        phi_changes_decision=scalar_split,
    )


def axes_from_items(items: list[dict[str, object]]) -> tuple[float, float, float]:
    """Journal axes: evidence opened, verified after a check, linkage."""

    if not items:
        return 0.0, 0.0, 0.0
    total = float(len(items))
    opened = sum(1 for item in items if item.get("evidence_opened") is True)
    verified = sum(1 for item in items if item.get("verified") is True)
    linked = sum(
        1
        for item in items
        if item.get("refs") or item.get("handoff_to")
    )
    return opened / total, verified / total, linked / total
