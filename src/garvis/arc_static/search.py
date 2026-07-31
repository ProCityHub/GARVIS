"""Deterministic program synthesis over the DSL (Track A, module 2).

Enumerates compositions of DSL primitives (depth 1 then depth 2) in a
fixed, deterministic order, and returns the FIRST program that reproduces
every training pair exactly. No program found means no answer — the solver
never guesses.

Governance (enforced by tests):
- Pure offline computation; deterministic given identical inputs.
- A returned program is VERIFIED: it maps every train input to its exact
  train output. Verification is not optional or samplable.
- Budgets are explicit (max nodes / max seconds); exhausting a budget is
  an honest, reported failure, not a fallback prediction.

Preregistered result at merge time (fixed harness, 0.5s/task budget):
ARC-1 public evaluation 2/400 pass@1 exact (prior baseline 1/400);
ARC-1 public training 28/400. Recorded whatever the numbers are.

Authorship: Adrien D. Thomas / ProCityHub. DIRECTIVE-014 module 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .dsl import (
    Grid,
    PARAM_FAMILIES,
    UNARY_PRIMITIVES,
    as_grid,
)

Step = Tuple[str, Callable[[Grid], Grid]]


class SearchError(ValueError):
    """Raised on invalid synthesis input."""


@dataclass(frozen=True)
class Program:
    names: Tuple[str, ...]
    steps: Tuple[Callable[[Grid], Grid], ...]

    def apply(self, g: Grid) -> Grid:
        out = as_grid(g)
        for fn in self.steps:
            out = fn(out)
        return out

    def __str__(self) -> str:
        return " -> ".join(self.names)


def _instruction_set() -> List[Step]:
    """Deterministic, ordered list of all candidate single steps."""
    steps: List[Step] = []
    for name in sorted(UNARY_PRIMITIVES):
        steps.append((name, UNARY_PRIMITIVES[name]))
    for fam in sorted(PARAM_FAMILIES):
        builder, params = PARAM_FAMILIES[fam]
        for p in params:
            label = f"{fam}{p}"
            steps.append((label, (lambda b, pp: lambda g: b(g, *pp))(builder, p)))
    return steps


def _fits(steps: Sequence[Step], pairs) -> bool:
    for inp, out in pairs:
        g = inp
        try:
            for _, fn in steps:
                g = fn(g)
        except Exception:
            return False
        if g != out:
            return False
    return True


def synthesize(
    train_pairs,
    max_depth: int = 2,
    max_nodes: int = 60_000,
    max_seconds: float = 20.0,
) -> Optional[Program]:
    """Find the first verified program, or None.

    Args:
        train_pairs: sequence of (input, output) grid-likes.
        max_depth: 1 or 2 composition steps.
        max_nodes: budget on candidate programs examined.
        max_seconds: wall-clock budget.
    """
    if max_depth not in (1, 2):
        raise SearchError("max_depth must be 1 or 2")
    pairs = [(as_grid(i), as_grid(o)) for i, o in train_pairs]
    if not pairs:
        raise SearchError("at least one training pair is required")

    instructions = _instruction_set()
    deadline = time.monotonic() + max_seconds
    examined = 0

    for step in instructions:
        examined += 1
        if examined > max_nodes or time.monotonic() > deadline:
            return None
        if _fits((step,), pairs):
            return Program((step[0],), (step[1],))

    if max_depth < 2:
        return None

    first_results = []
    for step in instructions:
        outs = []
        ok = True
        for inp, _ in pairs:
            try:
                outs.append(step[1](inp))
            except Exception:
                ok = False
                break
        if ok and any(o != inp for o, (inp, _) in zip(outs, pairs)):
            first_results.append((step, outs))

    for step_a, outs in first_results:
        for step_b in instructions:
            examined += 1
            if examined > max_nodes or time.monotonic() > deadline:
                return None
            good = True
            for mid, (_, target) in zip(outs, pairs):
                try:
                    if step_b[1](mid) != target:
                        good = False
                        break
                except Exception:
                    good = False
                    break
            if good:
                return Program(
                    (step_a[0], step_b[0]), (step_a[1], step_b[1])
                )
    return None


def solve_task(task: dict, **kwargs) -> Optional[List[Grid]]:
    """Synthesize from task['train']; if found, predict all test outputs."""
    if not isinstance(task, dict) or "train" not in task or "test" not in task:
        raise SearchError("task must contain 'train' and 'test'")
    program = synthesize(
        [(p["input"], p["output"]) for p in task["train"]], **kwargs
    )
    if program is None:
        return None
    return [program.apply(as_grid(t["input"])) for t in task["test"]]
