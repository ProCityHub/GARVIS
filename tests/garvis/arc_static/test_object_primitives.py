"""Tests for static ARC object-level primitives (Track A module 1b)."""

import pytest

from garvis.arc_static.dsl import UNARY_PRIMITIVES, as_grid
from garvis.arc_static.object_primitives import (
    ObjectPrimitiveError,
    crop_largest_object,
    extract_objects,
    filter_objects_by,
    keep_largest_object,
    largest_object,
    move_smallest_to_top_left,
    reconstruct_from_objects,
    reflect_largest_h,
    remove_smallest_object,
    smallest_object,
    translate_object,
)
from garvis.arc_static.search import synthesize


def test_extract_and_reconstruct_round_trip():
    grid = as_grid([[0, 2, 0, 3], [0, 2, 0, 3], [0, 0, 0, 0]])
    frame = extract_objects(grid)
    assert reconstruct_from_objects(frame) == grid
    assert [obj.color for obj in frame.objects] == [2, 3]


def test_filter_uses_measured_properties_only():
    frame = extract_objects([[4, 0, 0, 0], [4, 0, 5, 5], [0, 0, 0, 0]])
    assert [o.color for o in filter_objects_by(frame, colors={5})] == [5]
    assert [o.color for o in filter_objects_by(frame, min_pixels=2)] == [4, 5]
    assert [o.color for o in filter_objects_by(frame, touching_boundary=True)] == [4, 5]


def test_filter_rejects_invalid_pixel_range():
    frame = extract_objects([[0, 1]])
    with pytest.raises(ObjectPrimitiveError):
        filter_objects_by(frame, min_pixels=3, max_pixels=2)


def test_largest_and_smallest_are_deterministic():
    frame = extract_objects([[0, 2, 2, 0, 3], [0, 2, 0, 0, 0], [4, 0, 0, 0, 0]])
    largest = largest_object(frame)
    smallest = smallest_object(frame)
    assert largest is not None
    assert smallest is not None
    assert largest.color == 2
    assert smallest.color == 3
    assert smallest == smallest_object(frame)


def test_translation_preserves_shape_signature():
    frame = extract_objects([[0, 0, 0, 0], [0, 6, 6, 0], [0, 6, 0, 0]])
    obj = frame.objects[0]
    moved = translate_object(
        obj, dr=-1, dc=-1, height=frame.height, width=frame.width
    )
    assert moved.shape_signature == obj.shape_signature
    assert moved.bounding_box.min_row == 0
    assert moved.bounding_box.min_col == 0


def test_keep_and_crop_largest():
    grid = [[0, 2, 2, 0, 7], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert keep_largest_object(grid) == (
        (0, 2, 2, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    assert crop_largest_object(grid) == ((2, 2), (2, 0))


def test_remove_smallest():
    grid = [[0, 2, 2, 0, 7], [0, 2, 0, 0, 0]]
    assert remove_smallest_object(grid) == (
        (0, 2, 2, 0, 0),
        (0, 2, 0, 0, 0),
    )


def test_move_smallest_to_top_left_preserves_other_objects():
    grid = [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 0, 7]]
    assert move_smallest_to_top_left(grid) == (
        (7, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 2, 0, 0),
    )


def test_reflect_largest_h_inside_its_box():
    grid = [[0, 0, 0, 0], [0, 6, 6, 0], [0, 6, 0, 0]]
    assert reflect_largest_h(grid) == (
        (0, 0, 0, 0),
        (0, 6, 6, 0),
        (0, 0, 6, 0),
    )


def test_registered_primitives_are_grid_to_grid():
    expected = {
        "object_crop_largest",
        "object_crop_smallest",
        "object_keep_largest",
        "object_keep_smallest",
        "object_move_largest_top_left",
        "object_move_smallest_top_left",
        "object_reflect_largest_h",
        "object_reflect_largest_v",
        "object_remove_largest",
        "object_remove_smallest",
    }
    assert expected <= set(UNARY_PRIMITIVES)
    grid = as_grid([[0, 1], [0, 0]])
    for name in sorted(expected):
        out = UNARY_PRIMITIVES[name](grid)
        assert isinstance(out, tuple)
        assert out and all(isinstance(row, tuple) for row in out)


def test_search_can_synthesize_object_composite():
    train = [
        (
            [[0, 2, 2, 0, 7], [0, 2, 0, 0, 0]],
            [[0, 2, 2, 0, 0], [0, 2, 0, 0, 0]],
        ),
        (
            [[8, 0, 3, 3], [0, 0, 3, 0]],
            [[0, 0, 3, 3], [0, 0, 3, 0]],
        ),
    ]
    program = synthesize(train, max_depth=1)
    assert program is not None
    for input_grid, expected_grid in train:
        assert program.apply(as_grid(input_grid)) == as_grid(expected_grid)
    assert program.names[0] in {
        "object_keep_largest",
        "object_remove_smallest",
    }


def test_parser_remains_observational():
    frame = extract_objects(
        [[10, 10, 10, 10], [10, 12, 8, 14], [10, 10, 10, 10]],
        background=10,
    )
    assert {obj.color for obj in frame.objects} == {8, 12, 14}
    for obj in frame.objects:
        assert not hasattr(obj, "role")
        assert not hasattr(obj, "is_goal")
        assert not hasattr(obj, "is_wall")
        assert not hasattr(obj, "is_hazard")
