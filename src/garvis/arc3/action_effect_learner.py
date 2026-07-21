"""Deterministic ARC-3 action-effect learner (DIRECTIVE-011, module 3 of 6).

Learns, per action, what that action appears to do — purely from observed
FrameDiff evidence. Emits predictions with explicit confidence, and refuses
to predict below a minimum evidence threshold.

Governance (implementation facts, enforced by tests):
- Pure offline stdlib computation.
- Evidence-only: every prediction is backed by counted observations; the
  learner never invents effects and never predicts from zero or one sample
  (configurable, default minimum 2).
- Observational vocabulary only: effects are described as movement,
  appearance, vanishing, size/background change, or no change. No wall,
  hazard, goal, or player semantics — those belong to later modules.
- Deterministic: identical observation sequences yield identical models.

Authorship: Adrien D. Thomas / ProCityHub. Spec: GARVIS DIRECTIVE-011.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .frame_differ import FrameDiff


class ActionEffectError(ValueError):
    """Raised on invalid learner input."""


@dataclass(frozen=True)
class Effect:
    """Canonical observational summary of one action's outcome."""

    kind: str  # no_change | moved | appeared | vanished | mixed | frame_event
    details: Tuple = ()


@dataclass(frozen=True)
class Prediction:
    effect: Effect
    confidence: float       # modal share of observations, 0..1
    observations: int       # total observations for this action


def summarize_diff(diff: FrameDiff) -> Effect:
    """Deterministically reduce a FrameDiff to one canonical Effect."""
    if not isinstance(diff, FrameDiff):
        raise ActionEffectError("summarize_diff requires a FrameDiff")
    if diff.size_changed or diff.background_changed:
        return Effect(kind="frame_event",
                      details=(("size_changed", diff.size_changed),
                               ("background_changed", diff.background_changed)))
    moved = diff.moved
    appeared = diff.appeared
    vanished = diff.vanished
    changes = (len(moved) > 0) + (len(appeared) > 0) + (len(vanished) > 0)
    if changes == 0:
        return Effect(kind="no_change")
    if changes > 1 or len(moved) > 1:
        return Effect(kind="mixed",
                      details=(("moved", len(moved)),
                               ("appeared", len(appeared)),
                               ("vanished", len(vanished))))
    if moved:
        m = moved[0]
        return Effect(kind="moved", details=(("color", m.color),
                                             ("displacement", m.displacement)))
    if appeared:
        return Effect(kind="appeared", details=(("count", len(appeared)),))
    return Effect(kind="vanished", details=(("count", len(vanished)),))


@dataclass
class ActionEffectLearner:
    """Per-action evidence tally with confidence-scored prediction."""

    min_observations: int = 2
    _tallies: Dict[int, "Counter[Effect]"] = field(default_factory=dict)

    def observe(self, action_id: int, diff: FrameDiff) -> Effect:
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise ActionEffectError("action_id must be an integer")
        effect = summarize_diff(diff)
        self._tallies.setdefault(action_id, Counter())[effect] += 1
        return effect

    def observations(self, action_id: int) -> int:
        return sum(self._tallies.get(action_id, Counter()).values())

    def distribution(self, action_id: int) -> Tuple[Tuple[Effect, int], ...]:
        tally = self._tallies.get(action_id, Counter())
        # Deterministic order: count desc, then effect repr asc.
        return tuple(sorted(tally.items(), key=lambda kv: (-kv[1], repr(kv[0]))))

    def predict(self, action_id: int) -> Optional[Prediction]:
        """Modal effect with confidence, or None below evidence threshold."""
        total = self.observations(action_id)
        if total < self.min_observations:
            return None
        dist = self.distribution(action_id)
        effect, count = dist[0]
        return Prediction(effect=effect, confidence=count / total,
                          observations=total)

    def known_actions(self) -> Tuple[int, ...]:
        return tuple(sorted(self._tallies))

    def reset(self) -> None:
        """Discard all evidence (new game or new geometry)."""
        self._tallies.clear()
