"""Bounded working memory for GARVIS while preserving the complete archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window",
    "maximum context length",
    "input exceeds",
    "too many tokens",
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _serialized_size(item: Any) -> int:
    try:
        return len(json.dumps(item, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(item))


def _is_dialogue_item(item: Any) -> bool:
    return isinstance(item, dict) and (
        item.get("role") in {"user", "assistant", "system"}
        or item.get("type") == "message"
    )


def _contains_context_error(item: Any) -> bool:
    try:
        text = json.dumps(item, ensure_ascii=False, default=str).casefold()
    except (TypeError, ValueError):
        text = str(item).casefold()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


def bounded_session_input(history: list[Any], new_input: list[Any]) -> list[Any]:
    """Return recent safe dialogue plus the current turn."""
    total_budget = _env_int("GARVIS_CONTEXT_BUDGET_CHARS", 80_000)
    item_limit = _env_int("GARVIS_HISTORY_ITEM_LIMIT", 64)
    single_item_limit = _env_int("GARVIS_SINGLE_HISTORY_ITEM_CHARS", 16_000)

    remaining = max(
        0,
        total_budget - sum(_serialized_size(item) for item in new_input),
    )
    selected: list[Any] = []

    for item in reversed(history):
        if len(selected) >= item_limit:
            break
        if not _is_dialogue_item(item) or _contains_context_error(item):
            continue

        size = _serialized_size(item)
        if size > single_item_limit or size > remaining:
            continue

        selected.append(item)
        remaining -= size

    selected.reverse()
    return selected + list(new_input)


def make_bounded_session_input(
    model_input: str,
) -> Callable[[list[Any], list[Any]], list[Any]]:
    """Send enriched input to the model while storing only the clean user turn."""
    current_item = {"role": "user", "content": model_input}

    def callback(history: list[Any], _new_input: list[Any]) -> list[Any]:
        return bounded_session_input(history, [current_item])

    return callback


def _clip(message: str, limit: int, note: str) -> str:
    if len(message) <= limit:
        return message

    available = max(2_000, limit - len(note) - 4)
    head = available // 2
    tail = available - head
    return f"{message[:head]}\n\n{note}\n\n{message[-tail:]}"


def prepare_current_input(message: str, *, session_id: str) -> str:
    """Archive an oversized turn and return a bounded excerpt."""
    limit = _env_int("GARVIS_CURRENT_INPUT_CHARS", 32_000)
    if len(message) <= limit:
        return message

    root = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
    archive_dir = root / "oversized_inputs"
    archive_dir.mkdir(parents=True, exist_ok=True)

    safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id).strip("-") or "default"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
    path = archive_dir / f"{timestamp}-{safe_session}-{digest}.txt"
    path.write_text(message, encoding="utf-8")
    path.chmod(0o600)

    note = (
        "[GARVIS context governor: the complete input was preserved locally at "
        f"{path}; a bounded excerpt is being processed.]"
    )
    return _clip(message, limit, note)


def emergency_current_input(message: str) -> str:
    """Create a smaller retry request after a provider context rejection."""
    note = (
        "[GARVIS automatic recovery: old conversation history was omitted because "
        "the provider rejected the first request as too large.]"
    )
    return _clip(
        message,
        _env_int("GARVIS_EMERGENCY_INPUT_CHARS", 12_000),
        note,
    )


def is_context_length_error(error: BaseException) -> bool:
    """Recognize provider context-window errors."""
    text = str(error).casefold()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)
