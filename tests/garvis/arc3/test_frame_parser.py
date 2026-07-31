"""Tests for the deterministic ARC-3 frame parser (DIRECTIVE-011 module 1)."""

import dataclasses

import pytest

from garvis.arc3 import FrameValidationError, ParsedFrame, parse_frame


def test_single_cell_object():
    p = parse_frame([[0, 0], [0, 5]])
    assert p.background_color == 0
    assert len(p.objects) == 1
    o = p.objects[0]
    assert o.color == 5 and o.pixel_count == 1
    assert o.shape_signature == ((0, 0),)
    assert o.centroid == (1.0, 1.0)


def test_translated_copy_has_identical_signature():
    a = parse_frame([[7, 7, 0, 0], [0, 7, 0, 0]]).objects[0]
    b = parse_frame([[0, 0, 7, 7], [0, 0, 0, 7]]).objects[0]
    assert a.shape_signature == b.shape_signature
    assert a.bounding_box != b.bounding_box


def test_two_disconnected_same_color_objects():
    p = parse_frame([[3, 0, 3], [0, 0, 0]])
    assert len(p.objects) == 2
    assert all(o.color == 3 for o in p.objects)


def test_diagonal_cells_separate_under_4_connectivity():
    p = parse_frame([[2, 0], [0, 2]])
    assert len(p.objects) == 2


def test_diagonal_cells_joined_under_8_connectivity():
    p = parse_frame([[2, 0], [0, 2]], connectivity=8)
    assert len(p.objects) == 1
    assert p.objects[0].pixel_count == 2


def test_multiple_colors():
    p = parse_frame([[0, 1, 2], [0, 1, 0]])
    colors = sorted(o.color for o in p.objects)
    assert colors == [1, 2]
    assert p.color_count(1) == 2 and p.color_count(2) == 1


def test_explicit_background_override():
    p = parse_frame([[1, 1], [1, 2]], background=1)
    assert p.background_color == 1
    assert dict(p.metadata)["background_method"] == "override"
    assert len(p.objects) == 1 and p.objects[0].color == 2


def test_most_frequent_background_selection():
    p = parse_frame([[4, 4, 4], [4, 9, 4]])
    assert p.background_color == 4
    assert dict(p.metadata)["background_method"] == "most_frequent"


def test_deterministic_background_tie_break():
    # Colors 1 and 2 both appear twice: smaller color value wins.
    p = parse_frame([[1, 2], [2, 1]])
    assert p.background_color == 1


def test_non_zero_background():
    p = parse_frame([[6, 6, 6], [6, 0, 6]])
    assert p.background_color == 6
    assert len(p.objects) == 1 and p.objects[0].color == 0


def test_irregular_shape_bbox_and_centroid():
    p = parse_frame([
        [0, 5, 0],
        [5, 5, 0],
        [0, 0, 0],
    ])
    o = p.objects[0]
    assert (o.bounding_box.min_row, o.bounding_box.min_col) == (0, 0)
    assert (o.bounding_box.max_row, o.bounding_box.max_col) == (1, 1)
    assert o.bounding_box.height == 2 and o.bounding_box.width == 2
    assert o.centroid[0] == pytest.approx(2 / 3)
    assert o.centroid[1] == pytest.approx(2 / 3)


def test_boundary_contact():
    p = parse_frame([[5, 0, 0], [0, 0, 0], [0, 0, 3]], background=0)
    by_color = {o.color: o for o in p.objects}
    assert by_color[5].touches_boundary is True
    assert by_color[3].touches_boundary is True
    q = parse_frame([[0, 0, 0], [0, 7, 0], [0, 0, 0]])
    assert q.objects[0].touches_boundary is False


def test_ragged_input_rejected():
    with pytest.raises(FrameValidationError, match="ragged"):
        parse_frame([[1, 2], [1]])


def test_empty_input_rejected():
    with pytest.raises(FrameValidationError):
        parse_frame([])
    with pytest.raises(FrameValidationError):
        parse_frame([[]])


def test_non_integer_value_rejected():
    with pytest.raises(FrameValidationError, match="non-integer"):
        parse_frame([[1, "x"]])
    with pytest.raises(FrameValidationError, match="non-integer"):
        parse_frame([[1, 2.5]])
    with pytest.raises(FrameValidationError, match="non-integer"):
        parse_frame([[True, 1]])


def test_deterministic_object_ordering_and_ids():
    p = parse_frame([
        [0, 8, 0, 9],
        [7, 0, 0, 0],
    ])
    assert [o.object_id for o in p.objects] == [0, 1, 2]
    assert [o.color for o in p.objects] == [8, 9, 7]  # row-major order


def test_parser_assigns_no_semantic_roles():
    p = parse_frame([[0, 8], [10, 14]])
    fields = {f.name for f in dataclasses.fields(p.objects[0])}
    forbidden = {"role", "is_wall", "is_hazard", "is_goal", "is_player"}
    assert fields.isdisjoint(forbidden)
    assert not any(
        "wall" in k or "hazard" in k or "goal" in k or "player" in k
        for k, _ in p.metadata
    )


def test_practice_game_fixture_observational_only():
    # Colors from pc01: background 0, hazard 8, wall 10, self 12, goal 14 —
    # the parser must represent all of them purely observationally.
    frame = [
        [0, 0, 0, 0, 0],
        [0, 12, 0, 8, 0],
        [10, 10, 10, 10, 0],
        [0, 0, 0, 14, 0],
    ]
    p = parse_frame(frame)
    present = sorted({o.color for o in p.objects})
    assert present == [8, 10, 12, 14]
    wall_obj = next(o for o in p.objects if o.color == 10)
    goal_obj = next(o for o in p.objects if o.color == 14)
    assert wall_obj.shape_signature != goal_obj.shape_signature
    assert p.scale is None


def test_scale_is_none_for_ambiguous_input():
    assert parse_frame([[0, 1], [0, 0]]).scale is None


def test_repeat_parsing_is_structurally_equal():
    frame = [[0, 3, 0], [3, 3, 0], [0, 0, 6]]
    assert parse_frame(frame) == parse_frame(frame)
    assert isinstance(parse_frame(frame), ParsedFrame)


def test_numpy_input_accepted_when_available():
    np = pytest.importorskip("numpy")
    p = parse_frame(np.array([[0, 0], [0, 5]]))
    assert p.objects[0].color == 5
