"""Tests for the ARC-3 planner (DIRECTIVE-011 module 5)."""

import pytest

from garvis.arc3.planner import Plan, PlannerError, plan_path

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def test_straight_line_plan():
    p = plan_path((0, 0), (0, 3), MOVES, (1, 4))
    assert p.actions == (RIGHT, RIGHT, RIGHT)
    assert p.path[0] == (0, 0) and p.path[-1] == (0, 3)
    assert p.length == 3


def test_already_at_target():
    p = plan_path((2, 2), (2, 2), MOVES, (5, 5))
    assert p.actions == () and p.path == ((2, 2),)


def test_routes_around_forbidden_wall():
    # Wall column between start and target forces a detour.
    forbidden = frozenset({(0, 1), (1, 1)})
    p = plan_path((0, 0), (0, 2), MOVES, (3, 3), forbidden)
    assert p is not None
    assert not any(cell in forbidden for cell in p.path)
    assert p.length == 6  # down twice, right twice, up twice


def test_no_path_returns_none():
    forbidden = frozenset({(0, 1), (1, 0), (1, 1)})
    assert plan_path((0, 0), (2, 2), MOVES, (3, 3), forbidden) is None


def test_forbidden_target_refused():
    assert plan_path((0, 0), (0, 2), MOVES, (1, 3),
                     frozenset({(0, 2)})) is None


def test_forbidden_start_refused():
    assert plan_path((0, 0), (0, 2), MOVES, (1, 3),
                     frozenset({(0, 0)})) is None


def test_out_of_bounds_returns_none():
    assert plan_path((0, 0), (5, 5), MOVES, (3, 3)) is None
    assert plan_path((9, 9), (0, 0), MOVES, (3, 3)) is None


def test_limited_moves_respected():
    # Only RIGHT and DOWN known: target up-left is unreachable.
    limited = {RIGHT: (0, 1), DOWN: (1, 0)}
    assert plan_path((2, 2), (0, 0), limited, (3, 3)) is None
    p = plan_path((0, 0), (2, 2), limited, (3, 3))
    assert p is not None and p.length == 4


def test_deterministic_tie_break():
    # Two equal-length routes: plan must be identical across calls and
    # prefer lower action ids at each branch.
    a = plan_path((1, 1), (2, 2), MOVES, (4, 4))
    b = plan_path((1, 1), (2, 2), MOVES, (4, 4))
    assert a == b
    assert a.actions == (DOWN, RIGHT)  # DOWN(2) explored before RIGHT(4)


def test_non_unit_moves_supported():
    # Evidence said this game moves two cells per press: planner obeys.
    big = {RIGHT: (0, 2), DOWN: (2, 0)}
    p = plan_path((0, 0), (0, 4), big, (1, 5))
    assert p.actions == (RIGHT, RIGHT)


def test_rejects_bad_input():
    with pytest.raises(PlannerError):
        plan_path("start", (0, 0), MOVES, (3, 3))
    with pytest.raises(PlannerError):
        plan_path((0, 0), (1, 1), {}, (3, 3))
    with pytest.raises(PlannerError):
        plan_path((0, 0), (1, 1), {UP: (0, 0)}, (3, 3))
    with pytest.raises(PlannerError):
        plan_path((0, 0), (1, 1), MOVES, (0, 3))
    with pytest.raises(PlannerError):
        plan_path((0, 0), (1, 1), {True: (0, 1)}, (3, 3))


def test_plan_is_frozen_dataclass():
    p = plan_path((0, 0), (0, 1), MOVES, (1, 2))
    assert isinstance(p, Plan)
    with pytest.raises(Exception):
        p.actions = ()
