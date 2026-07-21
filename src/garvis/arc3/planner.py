"""Deterministic ARC-3 planner (DIRECTIVE-011, module 5 of 6).

Breadth-first pathfinding from a start cell to a target cell over the
game-cell grid, using only action moves the caller's evidence supports,
and never routing through forbidden cells.

Governance (implementation facts, enforced by tests):
- Pure offline stdlib computation.
- Evidence-only: the planner consumes (action_id -> (drow, dcol)) moves the
  caller derived from the action-effect learner; it invents no moves.
- Safety: forbidden cells are never entered, including the target; if the
  target is forbidden, planning fails explicitly.
- Honest failure: returns None when no path exists; never a guess.
- Deterministic: stable action ordering makes plans reproducible; ties in
  path length resolve by ascending action_id sequence.

Authorship: Adrien D. Thomas / ProCityHub. Spec: GARVIS DIRECTIVE-011.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

Cell = Tuple[int, int]  # (row, col)


class PlannerError(ValueError):
    """Raised on invalid planner input."""


@dataclass(frozen=True)
class Plan:
    actions: Tuple[int, ...]
    path: Tuple[Cell, ...]      # includes start and target

    @property
    def length(self) -> int:
        return len(self.actions)


def _check_cell(cell, name: str) -> Cell:
    if (
        not isinstance(cell, tuple)
        or len(cell) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in cell)
    ):
        raise PlannerError(f"{name} must be an (int, int) tuple")
    return cell


def plan_path(
    start: Cell,
    target: Cell,
    moves: Dict[int, Tuple[int, int]],
    grid_size: Tuple[int, int],
    forbidden: FrozenSet[Cell] = frozenset(),
) -> Optional[Plan]:
    """Shortest action sequence from start to target, or None.

    Args:
        start: current cell (row, col).
        target: destination cell (row, col).
        moves: evidence-backed map of action_id -> (drow, dcol). Must be
            non-empty; zero-displacement moves are rejected.
        grid_size: (height, width) bounds; cells outside are unreachable.
        forbidden: cells that may never be entered (walls, hazards).
    """
    _check_cell(start, "start")
    _check_cell(target, "target")
    if not isinstance(moves, dict) or not moves:
        raise PlannerError("moves must be a non-empty dict")
    ordered_moves = []
    for action_id in sorted(moves):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise PlannerError("action ids must be integers")
        delta = moves[action_id]
        _check_cell(delta, f"move for action {action_id}")
        if delta == (0, 0):
            raise PlannerError(f"action {action_id} has zero displacement")
        ordered_moves.append((action_id, delta))
    height, width = grid_size
    if height <= 0 or width <= 0:
        raise PlannerError("grid_size must be positive")

    def in_bounds(c: Cell) -> bool:
        return 0 <= c[0] < height and 0 <= c[1] < width

    if not in_bounds(start) or not in_bounds(target):
        return None
    if start in forbidden or target in forbidden:
        return None
    if start == target:
        return Plan(actions=(), path=(start,))

    queue = deque([start])
    came_from: Dict[Cell, Tuple[Cell, int]] = {}
    seen = {start}
    while queue:
        current = queue.popleft()
        for action_id, (dr, dc) in ordered_moves:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in seen or not in_bounds(nxt) or nxt in forbidden:
                continue
            seen.add(nxt)
            came_from[nxt] = (current, action_id)
            if nxt == target:
                actions = []
                path = [nxt]
                node = nxt
                while node != start:
                    prev, act = came_from[node]
                    actions.append(act)
                    path.append(prev)
                    node = prev
                actions.reverse()
                path.reverse()
                return Plan(actions=tuple(actions), path=tuple(path))
            queue.append(nxt)
    return None
