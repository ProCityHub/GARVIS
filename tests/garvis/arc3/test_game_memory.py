"""Tests for the ARC-3 per-game memory (DIRECTIVE-011 module 6)."""

import pytest

from garvis.arc3.game_memory import (
    GameMemory,
    GameMemoryError,
    has_memory,
    load_memory,
    save_memory,
)


def sample():
    return GameMemory(
        game_id="pc01",
        confirmed_goal_colors={14},
        demoted_colors={8},
        wall_colors={10},
        action_moves=((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))),
    )


def test_round_trip_structural_equality(tmp_path):
    m = sample()
    save_memory(m, str(tmp_path))
    loaded = load_memory(str(tmp_path), "pc01")
    assert loaded == m
    assert loaded.moves_dict()[4] == (0, 1)


def test_serialization_is_deterministic():
    assert sample().to_json() == sample().to_json()
    shuffled = GameMemory(
        game_id="pc01",
        confirmed_goal_colors={14},
        demoted_colors={8},
        wall_colors={10},
        action_moves=((4, (0, 1)), (1, (-1, 0)), (3, (0, -1)), (2, (1, 0))),
    )
    assert shuffled.to_json() == sample().to_json()


def test_has_memory(tmp_path):
    assert has_memory(str(tmp_path), "pc01") is False
    save_memory(sample(), str(tmp_path))
    assert has_memory(str(tmp_path), "pc01") is True


def test_missing_file_raises(tmp_path):
    with pytest.raises(GameMemoryError, match="no memory file"):
        load_memory(str(tmp_path), "ghost")


def test_corrupt_file_rejected(tmp_path):
    (tmp_path / "bad1.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(GameMemoryError, match="corrupt"):
        load_memory(str(tmp_path), "bad1")
    (tmp_path / "bad2.json").write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(GameMemoryError, match="JSON object"):
        load_memory(str(tmp_path), "bad2")


def test_wrong_schema_version_rejected(tmp_path):
    text = sample().to_json().replace('"schema_version":1', '"schema_version":99')
    (tmp_path / "pc01.json").write_text(text, encoding="utf-8")
    with pytest.raises(GameMemoryError, match="schema_version"):
        load_memory(str(tmp_path), "pc01")


def test_mismatched_game_id_rejected(tmp_path):
    save_memory(sample(), str(tmp_path))
    (tmp_path / "other.json").write_text(
        (tmp_path / "pc01.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(GameMemoryError, match="claims game_id"):
        load_memory(str(tmp_path), "other")


def test_overwrite_updates_cleanly(tmp_path):
    save_memory(sample(), str(tmp_path))
    updated = GameMemory(game_id="pc01", confirmed_goal_colors={14, 9})
    save_memory(updated, str(tmp_path))
    assert load_memory(str(tmp_path), "pc01") == updated


def test_empty_memory_round_trips(tmp_path):
    m = GameMemory(game_id="fresh")
    save_memory(m, str(tmp_path))
    loaded = load_memory(str(tmp_path), "fresh")
    assert loaded == m
    assert loaded.moves_dict() == {}


def test_rejects_bad_construction():
    with pytest.raises(GameMemoryError):
        GameMemory(game_id="")
    with pytest.raises(GameMemoryError):
        GameMemory(game_id="x", wall_colors={"ten"})
    with pytest.raises(GameMemoryError):
        GameMemory(game_id="x", demoted_colors={True})
    with pytest.raises(GameMemoryError):
        GameMemory(game_id="x", action_moves=((1, (0,)),))
    with pytest.raises(GameMemoryError):
        save_memory("not-a-memory", "/tmp")


def test_no_secret_like_fields():
    import dataclasses
    names = {f.name for f in dataclasses.fields(GameMemory)}
    assert names.isdisjoint({"api_key", "token", "password", "secret", "notes"})


def test_frozen_dataclass():
    m = sample()
    with pytest.raises(Exception):
        m.game_id = "hacked"
