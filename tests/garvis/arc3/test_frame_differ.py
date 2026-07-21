"""Tests for the deterministic ARC-3 frame differ (DIRECTIVE-011 module 2)."""

import pytest

from garvis.arc3.frame_differ import FrameDiffError, diff_frames
from garvis.arc3.frame_parser import parse_frame


def test_identical_frames():
    f = [[0, 5], [0, 0]]
    d = diff_frames(parse_frame(f), parse_frame(f))
    assert d.identical is True
    assert d.moved == () and d.appeared == () and d.vanished == ()
    assert len(d.unmoved) == 1


def test_single_object_moved_right():
    d = diff_frames(
        parse_frame([[5, 0, 0], [0, 0, 0]]),
        parse_frame([[0, 0, 5], [0, 0, 0]]),
    )
    assert d.identical is False
    assert len(d.moved) == 1
    assert d.moved[0].displacement == (0, 2)
    assert d.appeared == () and d.vanished == ()


def test_single_object_moved_down():
    d = diff_frames(
        parse_frame([[5, 0], [0, 0], [0, 0]]),
        parse_frame([[0, 0], [0, 0], [5, 0]]),
    )
    assert d.moved[0].displacement == (2, 0)


def test_object_appeared():
    d = diff_frames(
        parse_frame([[0, 0], [0, 0], [1, 1]], background=0),
        parse_frame([[0, 7], [0, 0], [1, 1]], background=0),
    )
    assert len(d.appeared) == 1 and d.vanished == ()
    assert len(d.unmoved) == 1  # the 1,1 pair stayed put


def test_object_vanished():
    d = diff_frames(
        parse_frame([[0, 7], [1, 1]], background=0),
        parse_frame([[0, 0], [1, 1]], background=0),
    )
    assert len(d.vanished) == 1 and d.appeared == ()


def test_shape_change_reports_vanish_plus_appear():
    # Same color, different shape: conservative matcher refuses the match.
    d = diff_frames(
        parse_frame([[5, 5, 0], [0, 0, 0]]),
        parse_frame([[5, 0, 0], [5, 0, 0]]),
    )
    assert len(d.vanished) == 1 and len(d.appeared) == 1
    assert d.matches == ()


def test_two_same_color_objects_nearest_matching():
    before = parse_frame([[3, 0, 0, 3], [0, 0, 0, 0]])
    after = parse_frame([[0, 3, 0, 3], [0, 0, 0, 0]])
    d = diff_frames(before, after)
    assert d.appeared == () and d.vanished == ()
    disp = sorted(m.displacement for m in d.matches)
    assert disp == [(0, 0), (0, 1)]  # right one stays, left one steps right


def test_multiple_objects_mixed_changes():
    d = diff_frames(
        parse_frame([[5, 0, 8], [0, 0, 0], [2, 0, 0]], background=0),
        parse_frame([[0, 5, 8], [0, 0, 0], [0, 0, 0]], background=0),
    )
    by_color = {m.color: m for m in d.matches}
    assert by_color[5].displacement == (0, 1)
    assert by_color[8].displacement == (0, 0)
    assert len(d.vanished) == 1  # color 2 gone


def test_size_change_flagged():
    d = diff_frames(
        parse_frame([[0, 5], [0, 0]]),
        parse_frame([[0, 5, 0], [0, 0, 0], [0, 0, 0]]),
    )
    assert d.size_changed is True and d.identical is False


def test_background_change_flagged():
    d = diff_frames(
        parse_frame([[0, 0, 5], [0, 0, 0]]),
        parse_frame([[7, 7, 5], [7, 7, 7]]),
    )
    assert d.background_changed is True and d.identical is False


def test_no_semantic_fields():
    import dataclasses
    d = diff_frames(parse_frame([[0, 5]]), parse_frame([[5, 0]]))
    names = {f.name for f in dataclasses.fields(d)}
    assert names.isdisjoint({"walls", "hazards", "goals", "player", "action"})


def test_deterministic_repeat():
    b = parse_frame([[3, 0, 3], [0, 8, 0]])
    a = parse_frame([[0, 3, 3], [8, 0, 0]])
    assert diff_frames(b, a) == diff_frames(b, a)


def test_rejects_non_parsed_input():
    with pytest.raises(FrameDiffError):
        diff_frames([[0, 1]], parse_frame([[0, 1]]))
