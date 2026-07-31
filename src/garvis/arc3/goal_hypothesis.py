"""Deterministic ARC-3 goal-hypothesis generator (DIRECTIVE-011, module 4 of 6).

Nominates candidate goal objects from a ParsedFrame and manages their
lifecycle strictly by game-outcome evidence:

    proposed  --(level completed while pursuing)-->  confirmed
    proposed  --(game over while pursuing)------->  demoted (hazard-like)

Governance (implementation facts, enforced by tests):
- Pure offline stdlib computation.
- A candidate is only ever a HYPOTHESIS until a level-completion event
  confirms it. Confirmation and demotion require explicit outcome evidence
  passed in by the caller; the generator never self-declares success.
- Demotion is per-color and permanent for the game unless explicitly reset.
- Deterministic nomination order (rarity, then position).
- No player/self detection here; the caller supplies colors to exclude.

Authorship: Adrien D. Thomas / ProCityHub. Spec: GARVIS DIRECTIVE-011.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

from .frame_parser import ParsedFrame


class GoalHypothesisError(ValueError):
    """Raised on invalid generator input."""


@dataclass(frozen=True)
class GoalCandidate:
    color: int
    position: Tuple[int, int]      # (row, col) bounding-box anchor
    pixel_count: int
    status: str                    # proposed | confirmed | demoted


@dataclass
class GoalHypothesisGenerator:
    """Per-game candidate nomination with evidence-driven lifecycle."""

    max_candidate_pixels: int = 4
    _confirmed: set = field(default_factory=set)   # colors
    _demoted: set = field(default_factory=set)     # colors

    # ----- lifecycle evidence (caller supplies outcomes) -----

    def record_level_completed(self, pursued_color: int) -> None:
        self._check_color(pursued_color)
        self._demoted.discard(pursued_color)
        self._confirmed.add(pursued_color)

    def record_game_over(self, pursued_color: int) -> None:
        self._check_color(pursued_color)
        if pursued_color in self._confirmed:
            # Conflicting evidence: keep both facts visible by removing
            # confirmation rather than silently preferring either.
            self._confirmed.discard(pursued_color)
        self._demoted.add(pursued_color)

    def confirmed_colors(self) -> FrozenSet[int]:
        return frozenset(self._confirmed)

    def demoted_colors(self) -> FrozenSet[int]:
        return frozenset(self._demoted)

    def reset(self) -> None:
        self._confirmed.clear()
        self._demoted.clear()

    # ----- nomination -----

    def candidates(
        self,
        frame: ParsedFrame,
        exclude_colors: Tuple[int, ...] = (),
    ) -> Tuple[GoalCandidate, ...]:
        """Nominate candidates from a frame, best-first.

        Confirmed colors rank first; then small rare objects. Demoted and
        excluded colors are never nominated. Background is always excluded.
        """
        if not isinstance(frame, ParsedFrame):
            raise GoalHypothesisError("candidates requires a ParsedFrame")
        for c in exclude_colors:
            self._check_color(c)
        excluded = set(exclude_colors) | {frame.background_color} | self._demoted

        out = []
        for obj in frame.objects:
            if obj.color in excluded:
                continue
            if obj.color in self._confirmed:
                status = "confirmed"
            elif obj.pixel_count <= self.max_candidate_pixels:
                status = "proposed"
            else:
                continue
            out.append(
                GoalCandidate(
                    color=obj.color,
                    position=(obj.bounding_box.min_row, obj.bounding_box.min_col),
                    pixel_count=obj.pixel_count,
                    status=status,
                )
            )
        # Deterministic best-first: confirmed first, then rarer (smaller),
        # then top-left position, then color.
        out.sort(
            key=lambda c: (
                0 if c.status == "confirmed" else 1,
                c.pixel_count,
                c.position,
                c.color,
            )
        )
        return tuple(out)

    @staticmethod
    def _check_color(color) -> None:
        if not isinstance(color, int) or isinstance(color, bool):
            raise GoalHypothesisError("color must be an integer")
