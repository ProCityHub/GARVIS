"""Tests for the ARC-3 goal-hypothesis generator (DIRECTIVE-011 module 4)."""

from typing import Any, cast

import pytest

from garvis.arc3.frame_parser import parse_frame
from garvis.arc3.goal_hypothesis import (
    GoalHypothesisError,
    GoalHypothesisGenerator,
)

FRAME = parse_frame([
    [0, 0, 0, 0, 0],
    [0, 12, 0, 8, 0],
    [10, 10, 10, 10, 10],
    [0, 0, 0, 14, 0],
], background=0)


def test_small_objects_nominated_large_ignored():
    g = GoalHypothesisGenerator()
    cands = g.candidates(FRAME, exclude_colors=(12,))
    colors = [c.color for c in cands]
    assert 8 in colors and 14 in colors
    assert 10 not in colors  # the 5-pixel bar exceeds the size cap
    assert all(c.status == "proposed" for c in cands)


def test_background_and_excluded_never_nominated():
    g = GoalHypothesisGenerator()
    colors = [c.color for c in g.candidates(FRAME, exclude_colors=(12, 8))]
    assert 0 not in colors and 12 not in colors and 8 not in colors


def test_all_candidates_start_as_hypotheses():
    g = GoalHypothesisGenerator()
    assert all(c.status == "proposed" for c in g.candidates(FRAME))
    assert g.confirmed_colors() == frozenset()
    assert g.demoted_colors() == frozenset()


def test_game_over_demotes_candidate():
    g = GoalHypothesisGenerator()
    g.record_game_over(8)
    colors = [c.color for c in g.candidates(FRAME, exclude_colors=(12,))]
    assert 8 not in colors
    assert g.demoted_colors() == frozenset({8})


def test_level_completed_confirms_candidate():
    g = GoalHypothesisGenerator()
    g.record_level_completed(14)
    cands = g.candidates(FRAME, exclude_colors=(12,))
    by_color = {c.color: c for c in cands}
    assert by_color[14].status == "confirmed"
    assert g.confirmed_colors() == frozenset({14})


def test_confirmed_ranks_before_proposed():
    g = GoalHypothesisGenerator()
    g.record_level_completed(14)
    cands = g.candidates(FRAME, exclude_colors=(12,))
    assert cands[0].color == 14 and cands[0].status == "confirmed"


def test_practice_game_sequence_reproduces_fix():
    # The exact PC01 lesson: chase 8 -> die -> demote; chase 14 -> level
    # complete -> confirm; later frames nominate 14 first and 8 never.
    g = GoalHypothesisGenerator()
    first = [c.color for c in g.candidates(FRAME, exclude_colors=(12,))]
    assert set(first) == {8, 14}
    g.record_game_over(8)
    g.record_level_completed(14)
    after = g.candidates(FRAME, exclude_colors=(12,))
    assert [c.color for c in after] == [14]
    assert after[0].status == "confirmed"


def test_conflicting_evidence_removes_confirmation():
    g = GoalHypothesisGenerator()
    g.record_level_completed(14)
    g.record_game_over(14)
    assert 14 in g.demoted_colors()
    assert 14 not in g.confirmed_colors()


def test_deterministic_ordering():
    frame = parse_frame([
        [0, 7, 0, 9],
        [0, 0, 0, 0],
        [3, 0, 0, 0],
    ], background=0)
    g = GoalHypothesisGenerator()
    a = g.candidates(frame)
    b = g.candidates(frame)
    assert a == b
    # All single-pixel: order by top-left position.
    assert [c.color for c in a] == [7, 9, 3]


def test_reset_clears_lifecycle():
    g = GoalHypothesisGenerator()
    g.record_game_over(8)
    g.record_level_completed(14)
    g.reset()
    assert g.confirmed_colors() == frozenset()
    assert g.demoted_colors() == frozenset()


def test_rejects_bad_input():
    g = GoalHypothesisGenerator()
    with pytest.raises(GoalHypothesisError):
        g.candidates(cast(Any, [[0, 1]]))
    with pytest.raises(GoalHypothesisError):
        g.record_game_over(cast(Any, "hazard"))
    with pytest.raises(GoalHypothesisError):
        g.record_level_completed(True)


def test_size_cap_configurable():
    g = GoalHypothesisGenerator(max_candidate_pixels=5)
    colors = [c.color for c in g.candidates(FRAME, exclude_colors=(12,))]
    assert 10 in colors  # bar admitted under the larger cap
