"""Tests for the bounded conversation session (context-window guard)."""

import pytest

from garvis.bounded_session import BoundedSession


class FakeInner:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.cleared = False

    async def get_items(self, limit=None):
        return list(self.items) if limit is None else list(self.items[-limit:])

    async def add_items(self, items):
        self.items.extend(items)

    async def pop_item(self):
        return self.items.pop() if self.items else None

    async def clear_session(self):
        self.items = []
        self.cleared = True


def msg(i, size=10):
    return {"role": "user", "content": f"m{i}:" + "x" * size}


@pytest.mark.asyncio
async def test_reads_bounded_to_most_recent_items():
    inner = FakeInner([msg(i) for i in range(500)])
    s = BoundedSession(inner, max_items=40, max_chars=1_000_000)
    out = await s.get_items()
    assert len(out) == 40
    assert out[0]["content"].startswith("m460:")
    assert out[-1]["content"].startswith("m499:")


@pytest.mark.asyncio
async def test_character_budget_drops_oldest_first():
    inner = FakeInner([
        {"c": "a" * 90_000},
        {"c": "b" * 90_000},
        {"c": "tail"},
    ])
    s = BoundedSession(inner, max_items=40, max_chars=120_000)
    out = await s.get_items()
    assert out[-1]["c"] == "tail"
    assert len(out) == 2


@pytest.mark.asyncio
async def test_single_oversized_newest_item_still_returned():
    inner = FakeInner([{"c": "giant" * 100_000}])
    s = BoundedSession(inner, max_items=40, max_chars=1_000)
    assert len(await s.get_items()) == 1


@pytest.mark.asyncio
async def test_caller_limit_respected_and_clamped():
    inner = FakeInner([msg(i) for i in range(100)])
    s = BoundedSession(inner, max_items=40, max_chars=1_000_000)
    assert len(await s.get_items(limit=5)) == 5
    assert len(await s.get_items(limit=999)) == 40


@pytest.mark.asyncio
async def test_writes_delegate_and_nothing_deleted_on_read():
    inner = FakeInner([msg(i) for i in range(100)])
    s = BoundedSession(inner, max_items=10, max_chars=1_000_000)
    await s.get_items()
    assert len(inner.items) == 100
    await s.add_items([msg(100)])
    assert len(inner.items) == 101
    await s.clear_session()
    assert inner.cleared is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GARVIS_MAX_HISTORY_ITEMS", "7")
    monkeypatch.setenv("GARVIS_MAX_HISTORY_CHARS", "5000")
    s = BoundedSession(FakeInner())
    assert s.max_items == 7 and s.max_chars == 5000


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        BoundedSession(None)
    with pytest.raises(ValueError):
        BoundedSession(FakeInner(), max_items=-1)


@pytest.mark.asyncio
async def test_default_factory_returns_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("GARVIS_MAX_HISTORY_ITEMS", raising=False)
    monkeypatch.delenv("GARVIS_MAX_HISTORY_CHARS", raising=False)
    from garvis.assistant import _default_session_factory
    s = _default_session_factory("test", tmp_path / "s.db")
    assert isinstance(s, BoundedSession)
