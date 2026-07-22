"""Tests for the ARC-3 action-effect learner (DIRECTIVE-011 module 3)."""

import dataclasses
from typing import Any, cast

import pytest

from garvis.arc3.action_effect_learner import (
    ActionEffectError,
    ActionEffectLearner,
    Effect,
    summarize_diff,
)
from garvis.arc3.frame_differ import diff_frames
from garvis.arc3.frame_parser import parse_frame


def d(before, after, **kw):
    return diff_frames(parse_frame(before, **kw), parse_frame(after, **kw))


def test_summarize_no_change():
    e = summarize_diff(d([[0, 5]], [[0, 5]]))
    assert e == Effect(kind="no_change")


def test_summarize_single_move():
    e = summarize_diff(d([[5, 0, 0]], [[0, 5, 0]]))
    assert e.kind == "moved"
    assert dict(e.details)["displacement"] == (0, 1)
    assert dict(e.details)["color"] == 5


def test_summarize_appear_and_vanish():
    assert summarize_diff(
        d([[0, 0], [1, 1]], [[7, 0], [1, 1]], background=0)
    ).kind == "appeared"
    assert summarize_diff(
        d([[7, 0], [1, 1]], [[0, 0], [1, 1]], background=0)
    ).kind == "vanished"


def test_summarize_mixed():
    e = summarize_diff(
        d([[5, 0, 8], [2, 0, 0]], [[0, 5, 8], [0, 0, 0]], background=0)
    )
    assert e.kind == "mixed"


def test_summarize_frame_event():
    e = summarize_diff(d([[0, 5]], [[0, 5, 0], [0, 0, 0]]))
    assert e.kind == "frame_event"


def test_summarize_rejects_non_diff():
    with pytest.raises(ActionEffectError):
        summarize_diff(cast(Any, "nope"))


def test_no_prediction_below_threshold():
    learner = ActionEffectLearner()
    learner.observe(1, d([[5, 0]], [[0, 5]]))
    assert learner.predict(1) is None  # one sample is not evidence enough


def test_prediction_with_full_confidence():
    learner = ActionEffectLearner()
    for _ in range(3):
        learner.observe(1, d([[5, 0]], [[0, 5]]))
    p = learner.predict(1)
    assert p is not None
    assert p.effect.kind == "moved"
    assert p.confidence == 1.0 and p.observations == 3


def test_prediction_confidence_is_modal_share():
    learner = ActionEffectLearner()
    learner.observe(2, d([[5, 0]], [[0, 5]]))
    learner.observe(2, d([[5, 0]], [[0, 5]]))
    learner.observe(2, d([[0, 5]], [[0, 5]]))  # blocked once
    p = learner.predict(2)
    assert p is not None
    assert p.effect.kind == "moved"
    assert p.confidence == pytest.approx(2 / 3)
    assert p.observations == 3


def test_actions_learned_independently():
    learner = ActionEffectLearner()
    for _ in range(2):
        learner.observe(1, d([[5, 0]], [[0, 5]]))
        learner.observe(3, d([[0, 5]], [[0, 5]]))
    first = learner.predict(1)
    third = learner.predict(3)
    assert first is not None and third is not None
    assert first.effect.kind == "moved"
    assert third.effect.kind == "no_change"
    assert learner.known_actions() == (1, 3)


def test_unknown_action_predicts_none():
    assert ActionEffectLearner().predict(6) is None


def test_distribution_is_deterministic():
    a, b = ActionEffectLearner(), ActionEffectLearner()
    seq = [d([[5, 0]], [[0, 5]]), d([[0, 5]], [[0, 5]]), d([[5, 0]], [[0, 5]])]
    for x in seq:
        a.observe(4, x)
        b.observe(4, x)
    assert a.distribution(4) == b.distribution(4)
    assert a.predict(4) == b.predict(4)


def test_reset_clears_evidence():
    learner = ActionEffectLearner()
    for _ in range(2):
        learner.observe(1, d([[5, 0]], [[0, 5]]))
    learner.reset()
    assert learner.predict(1) is None
    assert learner.known_actions() == ()


def test_rejects_bad_action_id():
    learner = ActionEffectLearner()
    with pytest.raises(ActionEffectError):
        learner.observe(cast(Any, "ACTION1"), d([[0, 5]], [[0, 5]]))
    with pytest.raises(ActionEffectError):
        learner.observe(True, d([[0, 5]], [[0, 5]]))


def test_no_semantic_fields():
    learner = ActionEffectLearner()
    for _ in range(2):
        learner.observe(1, d([[5, 0]], [[0, 5]]))
    p = learner.predict(1)
    assert p is not None
    names = {f.name for f in dataclasses.fields(p)}
    assert names.isdisjoint({"wall", "hazard", "goal", "player", "role"})
