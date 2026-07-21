"""Tests for the static-ARC program synthesis engine (Track A module 2)."""

import pytest

from garvis.arc_static.dsl import as_grid, rot90, replace_color
from garvis.arc_static.search import Program, SearchError, solve_task, synthesize


def pairs_from(fn, inputs):
    return [(g, fn(as_grid(g))) for g in inputs]


def test_depth1_synthesis():
    train = pairs_from(rot90, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    p = synthesize(train)
    assert p is not None and p.names == ("rot90",)
    assert p.apply(as_grid([[9, 0], [1, 2]])) == rot90(as_grid([[9, 0], [1, 2]]))


def test_depth2_synthesis():
    fn = lambda g: replace_color(rot90(g), 1, 5)
    train = pairs_from(fn, [[[1, 2], [3, 4]], [[1, 1], [0, 2]]])
    p = synthesize(train)
    assert p is not None and len(p.names) == 2
    assert p.apply(as_grid([[1, 0], [1, 3]])) == fn(as_grid([[1, 0], [1, 3]]))


def test_verified_on_all_pairs_or_none():
    train = [
        ([[1, 2], [3, 4]], rot90(as_grid([[1, 2], [3, 4]]))),
        ([[1, 2], [3, 4]], as_grid([[9, 9], [9, 9]])),
    ]
    assert synthesize(train) is None


def test_deterministic_program_choice():
    train = pairs_from(rot90, [[[1, 2], [3, 4]]])
    a, b = synthesize(train), synthesize(train)
    assert a.names == b.names


def test_budgets_are_honest():
    train = [([[1, 2], [3, 4]], [[7, 7, 7], [7, 7, 7], [7, 7, 7]])]
    assert synthesize(train, max_nodes=5) is None
    assert synthesize(train, max_seconds=0.0) is None


def test_rejects_bad_input():
    with pytest.raises(SearchError):
        synthesize([], max_depth=2)
    with pytest.raises(SearchError):
        synthesize([([[1]], [[1]])], max_depth=3)
    with pytest.raises(SearchError):
        solve_task({"train": []})


def test_solve_task_predicts_all_tests():
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},
        ],
        "test": [
            {"input": [[0, 1], [2, 3]]},
            {"input": [[9, 9], [1, 2]]},
        ],
    }
    preds = solve_task(task)
    assert preds is not None and len(preds) == 2
    assert preds[0] == ((2, 0), (3, 1))


def test_solve_task_no_program_returns_none():
    task = {
        "train": [{"input": [[1]], "output": [[1, 2, 3], [4, 5, 6]]}],
        "test": [{"input": [[2]]}],
    }
    assert solve_task(task, max_seconds=1.0) is None


def test_program_str():
    train = pairs_from(rot90, [[[1, 2], [3, 4]]])
    assert "rot90" in str(synthesize(train))
