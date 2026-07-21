"""Tests for the static-ARC DSL primitives (Track A module 1)."""

import pytest

from garvis.arc_static import dsl
from garvis.arc_static.dsl import (
    DslError,
    UNARY_PRIMITIVES,
    PARAM_FAMILIES,
    as_grid,
)

G = ((1, 2), (3, 4))


def test_as_grid_validates():
    assert as_grid([[1, 2], [3, 4]]) == G
    with pytest.raises(DslError):
        as_grid([])
    with pytest.raises(DslError):
        as_grid([[1], [1, 2]])
    with pytest.raises(DslError):
        as_grid([[1, "x"]])
    with pytest.raises(DslError):
        as_grid([[True, 1]])


def test_rotations_and_flips():
    assert dsl.rot90(G) == ((3, 1), (4, 2))
    assert dsl.rot180(G) == ((4, 3), (2, 1))
    assert dsl.rot270(G) == ((2, 4), (1, 3))
    assert dsl.flip_h(G) == ((2, 1), (4, 3))
    assert dsl.flip_v(G) == ((3, 4), (1, 2))
    assert dsl.transpose(G) == ((1, 3), (2, 4))
    assert dsl.rot90(dsl.rot270(G)) == G


def test_scaling_round_trip():
    up = dsl.upscale(G, 2)
    assert up == ((1, 1, 2, 2), (1, 1, 2, 2), (3, 3, 4, 4), (3, 3, 4, 4))
    assert dsl.downscale(up, 2) == G
    with pytest.raises(DslError):
        dsl.downscale(G, 3)


def test_tile_and_concat():
    assert dsl.tile(G, 1, 2) == ((1, 2, 1, 2), (3, 4, 3, 4))
    assert dsl.hconcat(G, G) == ((1, 2, 1, 2), (3, 4, 3, 4))
    assert dsl.vconcat(G, G) == ((1, 2), (3, 4), (1, 2), (3, 4))
    with pytest.raises(DslError):
        dsl.hconcat(G, ((1, 2),))


def test_color_ops():
    assert dsl.replace_color(G, 1, 9) == ((9, 2), (3, 4))
    assert dsl.swap_colors(G, 1, 4) == ((4, 2), (3, 1))
    assert dsl.keep_color(G, 3, fill=0) == ((0, 0), (3, 0))
    assert dsl.remove_color(G, 3, fill=0) == ((1, 2), (0, 4))
    assert dsl.map_colors(G, {1: 5, 4: 6}) == ((5, 2), (3, 6))


def test_shift_and_pad_and_border():
    assert dsl.shift(G, 0, 1, fill=0) == ((0, 1), (0, 3))
    assert dsl.pad(G, 1, fill=7)[0] == (7, 7, 7, 7)
    b = dsl.border(dsl.pad(G, 1, fill=0), 8)
    assert b[0] == (8, 8, 8, 8) and b[1][1] == 1


def test_crop_to_content():
    g = ((0, 0, 0, 0), (0, 5, 6, 0), (0, 0, 0, 0))
    assert dsl.crop_to_content(g) == ((5, 6),)
    all_bg = ((0, 0), (0, 0))
    assert dsl.crop_to_content(all_bg) == all_bg


def test_gravity_down():
    g = ((5, 0), (0, 0), (0, 6))
    assert dsl.gravity_down(g) == ((0, 0), (0, 0), (5, 6))


def test_background_color_deterministic_tie():
    assert dsl.background_color(((1, 2), (2, 1))) == 1


def test_every_unary_primitive_is_pure_and_valid():
    src = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    frozen = as_grid(src)
    for name, fn in UNARY_PRIMITIVES.items():
        out = fn(frozen)
        assert isinstance(out, tuple) and isinstance(out[0], tuple), name
        widths = {len(r) for r in out}
        assert len(widths) == 1, f"{name} produced ragged grid"
        assert frozen == as_grid(src), f"{name} mutated its input"
        assert fn(frozen) == out, f"{name} is non-deterministic"


def test_param_families_enumerable_and_valid():
    g = as_grid([[0, 1], [2, 3]])
    total = 0
    for name, (builder, params) in PARAM_FAMILIES.items():
        for p in params[:20]:
            try:
                out = builder(g, *p)
            except DslError:
                continue
            assert isinstance(out, tuple), name
            total += 1
    assert total > 0
    assert dsl.registry_size() > 100


def test_mirror_composites():
    assert UNARY_PRIMITIVES["mirror_right"](G) == ((1, 2, 2, 1), (3, 4, 4, 3))
    assert UNARY_PRIMITIVES["mirror_down"](G) == ((1, 2), (3, 4), (3, 4), (1, 2))
