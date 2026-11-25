"""memoryHints retrieval.

Providers return at most four hints. Empty fields are omitted. A contradicted
memory is not re-served as verified. The Memori SDK stub does not open a
network connection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .redact import redact_text

MAX_HINTS = 4
HINT_WEIGHT = 0.05
HINT_CAP = 0.20


@dataclass(frozen=True)
class MemoryHint:
    text: str
    evidence: str | None = None
    verified: bool | None = None
    source_id: str | None = None
    refs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.text:
            payload["text"] = self.text
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.verified is not None:
            payload["verified"] = self.verified
        if self.source_id:
            payload["source_id"] = self.source_id
        if self.refs:
            payload["refs"] = list(self.refs)
        return payload


@dataclass(frozen=True)
class StoredMemory:
    text: str
    evidence: str | None = None
    verified: bool | None = None
    source_id: str | None = None
    refs: tuple[str, ...] = ()
    contradicted: bool = False


def hint_contribution(hints: list[dict[str, object]]) -> float:
    """Bridge credit: 0.05 per non-empty hint, cap 0.20. Echo is not stamped."""

    nonempty = sum(1 for hint in hints if str(hint.get("text") or "").strip())
    return min(HINT_CAP, HINT_WEIGHT * nonempty)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


class InMemoryHintProvider:
    """Fixture store. Ranking is token overlap, not a phi score."""

    def __init__(self, memories: list[StoredMemory] | tuple[StoredMemory, ...] = ()):
        self.memories = list(memories)
        self.last_withheld = 0
        self.network_calls = 0

    def retrieve(self, query: str, *, max_hints: int = MAX_HINTS) -> list[MemoryHint]:
        limit = max(0, min(MAX_HINTS, int(max_hints)))
        query_tokens = _tokens(query)
        ranked: list[tuple[int, str, StoredMemory]] = []
        withheld = 0
        for memory in self.memories:
            if memory.contradicted:
                withheld += 1
                continue
            overlap = len(query_tokens & _tokens(memory.text))
            if overlap <= 0:
                continue
            ranked.append((overlap, memory.source_id or "", memory))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        self.last_withheld = withheld
        hints: list[MemoryHint] = []
        for _overlap, _source, memory in ranked:
            text = redact_text(memory.text).strip()
            if not text:
                continue
            evidence = redact_text(memory.evidence).strip() if memory.evidence else None
            hints.append(
                MemoryHint(
                    text=text,
                    evidence=evidence or None,
                    verified=memory.verified,
                    source_id=memory.source_id,
                    refs=memory.refs,
                )
            )
            if len(hints) >= limit:
                break
        return hints


class MemoriSdkHintProvider:
    """Stub for a later Memori SDK adapter.

    ``retrieve`` returns no hints and does not import or call the SDK.
    """

    def __init__(self) -> None:
        self.last_withheld = 0
        self.network_calls = 0
        self.sdk_calls = 0

    def retrieve(self, query: str, *, max_hints: int = MAX_HINTS) -> list[MemoryHint]:
        del query, max_hints
        self.last_withheld = 0
        return []
