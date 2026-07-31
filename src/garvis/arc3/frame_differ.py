"""Deterministic ARC-3 frame differ (DIRECTIVE-011, module 2 of 6).

Compares two ParsedFrame results and reports observational changes:
which objects persisted (and their displacement), appeared, or vanished.

Governance (implementation facts, enforced by tests):
- Pure offline stdlib computation.
- Observational only: no semantic roles, no action attribution, no causal
  claims. "What changed" only; "why" belongs to the action-effect learner.
- Deterministic: identical inputs yield structurally equal outputs.
- Matching is conservative: objects match only when color AND shape
  signature are identical; ambiguity is resolved by nearest displacement
  with deterministic tie-breaking. Unmatched objects are reported, never
  force-matched.

Authorship: Adrien D. Thomas / ProCityHub. Spec: GARVIS DIRECTIVE-011.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .frame_parser import ArcObject, ParsedFrame


class FrameDiffError(ValueError):
    """Raised when frames cannot be compared."""


@dataclass(frozen=True)
class ObjectMatch:
    """A persisted object: same color and shape in both frames."""

    before_id: int
    after_id: int
    color: int
    displacement: Tuple[int, int]  # (delta_row, delta_col), 0,0 = unmoved


@dataclass(frozen=True)
class FrameDiff:
    identical: bool
    size_changed: bool
    matches: Tuple[ObjectMatch, ...]
    appeared: Tuple[int, ...]   # after-frame object_ids with no match
    vanished: Tuple[int, ...]   # before-frame object_ids with no match
    background_changed: bool

    @property
    def moved(self) -> Tuple[ObjectMatch, ...]:
        return tuple(m for m in self.matches if m.displacement != (0, 0))

    @property
    def unmoved(self) -> Tuple[ObjectMatch, ...]:
        return tuple(m for m in self.matches if m.displacement == (0, 0))


def _anchor(obj: ArcObject) -> Tuple[int, int]:
    return (obj.bounding_box.min_row, obj.bounding_box.min_col)


def diff_frames(before: ParsedFrame, after: ParsedFrame) -> FrameDiff:
    """Compute an observational diff between two parsed frames."""
    if not isinstance(before, ParsedFrame) or not isinstance(after, ParsedFrame):
        raise FrameDiffError("diff_frames requires two ParsedFrame inputs")

    size_changed = (before.height, before.width) != (after.height, after.width)
    background_changed = before.background_color != after.background_color
    identical = (
        not size_changed
        and not background_changed
        and before.color_counts == after.color_counts
        and tuple((o.color, o.shape_signature, _anchor(o)) for o in before.objects)
        == tuple((o.color, o.shape_signature, _anchor(o)) for o in after.objects)
    )

    unmatched_after = list(after.objects)
    matches = []
    vanished = []
    for b in before.objects:
        candidates = [
            a for a in unmatched_after
            if a.color == b.color and a.shape_signature == b.shape_signature
        ]
        if not candidates:
            vanished.append(b.object_id)
            continue
        br, bc = _anchor(b)
        # Nearest displacement; ties broken by (row, col, object_id).
        best = min(
            candidates,
            key=lambda a: (
                abs(_anchor(a)[0] - br) + abs(_anchor(a)[1] - bc),
                _anchor(a)[0],
                _anchor(a)[1],
                a.object_id,
            ),
        )
        ar, ac = _anchor(best)
        matches.append(
            ObjectMatch(
                before_id=b.object_id,
                after_id=best.object_id,
                color=b.color,
                displacement=(ar - br, ac - bc),
            )
        )
        unmatched_after.remove(best)

    appeared = tuple(a.object_id for a in unmatched_after)
    return FrameDiff(
        identical=identical,
        size_changed=size_changed,
        matches=tuple(matches),
        appeared=appeared,
        vanished=tuple(vanished),
        background_changed=background_changed,
    )
