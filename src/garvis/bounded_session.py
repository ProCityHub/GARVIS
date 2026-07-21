"""Bounded conversation session for GARVIS.

Wraps any session (typically SQLiteSession) so that reads return only the
most recent history that fits a configured window, while writes preserve
the FULL history on disk untouched.

Why this exists (implementation fact): the agent runner requests the entire
session history every turn (`session.get_items()` with no limit). Long-lived
sessions eventually exceed the model's context window and every request
fails with `context_length_exceeded`. This wrapper guarantees the model is
fed a bounded, most-recent slice regardless of total history size.

Guarantees (enforced by tests):
- Nothing is ever deleted: add/pop/clear delegate verbatim; the underlying
  database keeps the complete record.
- Reads are bounded by BOTH an item cap and a character budget, newest
  first, returned in chronological order.
- An explicit smaller `limit` from a caller is respected; a larger one is
  clamped to the configured cap.
- Defaults are conservative and overridable via environment variables:
  GARVIS_MAX_HISTORY_ITEMS (default 40)
  GARVIS_MAX_HISTORY_CHARS (default 120000)

Authorship: Adrien D. Thomas / ProCityHub.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

DEFAULT_MAX_ITEMS = 40
DEFAULT_MAX_CHARS = 120_000


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _item_size(item: Any) -> int:
    try:
        return len(json.dumps(item, default=str))
    except (TypeError, ValueError):
        return len(str(item))


class BoundedSession:
    """Delegating session wrapper with bounded reads."""

    def __init__(
        self,
        inner: Any,
        max_items: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> None:
        if inner is None:
            raise ValueError("BoundedSession requires an inner session")
        self._inner = inner
        self.max_items = max_items or _env_int(
            "GARVIS_MAX_HISTORY_ITEMS", DEFAULT_MAX_ITEMS)
        self.max_chars = max_chars or _env_int(
            "GARVIS_MAX_HISTORY_CHARS", DEFAULT_MAX_CHARS)
        if self.max_items <= 0 or self.max_chars <= 0:
            raise ValueError("bounds must be positive")

    async def get_items(self, limit: Optional[int] = None) -> List[Any]:
        cap = self.max_items if limit is None else max(1, min(limit, self.max_items))
        items = await self._inner.get_items(limit=cap)
        kept: List[Any] = []
        used = 0
        for item in reversed(items):
            size = _item_size(item)
            if kept and used + size > self.max_chars:
                break
            kept.append(item)
            used += size
            if len(kept) >= cap:
                break
        kept.reverse()
        return kept

    async def add_items(self, items: List[Any]) -> None:
        await self._inner.add_items(items)

    async def pop_item(self) -> Optional[Any]:
        return await self._inner.pop_item()

    async def clear_session(self) -> None:
        await self._inner.clear_session()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
