"""Post-response consolidation into episodic and semantic memory."""

from __future__ import annotations

import re

from .models import EvidenceStatus, MemoryKind
from .segmentation import segment
from .store import NeuroStore

_DURABLE_PATTERNS = (
    re.compile(r"\b(?:i|we)\s+(?:want|prefer|need|own|created|built|use)\b", re.I),
    re.compile(r"\b(?:my|our)\s+(?:name|project|company|system|framework|email)\b", re.I),
)


def _semantic_candidates(user_text: str) -> tuple[str, ...]:
    signal = segment(user_text)
    candidates: list[str] = []
    for sentence in signal.sentences:
        if any(pattern.search(sentence) for pattern in _DURABLE_PATTERNS):
            candidates.append(sentence[:2_000])
    return tuple(dict.fromkeys(candidates))


def _existing_contents(store: NeuroStore, session_id: str) -> set[str]:
    return {
        memory.content
        for memory in store.list_memories(session_id=session_id, limit=1_000)
    }


def consolidate_turn(
    store: NeuroStore,
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    # The exact pair remains in the immutable episode archive.
    store.add_episode(session_id, user_text, assistant_text)

    # Searchable episodic memory indexes the user's experience, not GARVIS's own
    # generated prose. This prevents self-echo and response amplification.
    episode_memory = f"User request: {user_text[:2_500]}"
    existing = _existing_contents(store, session_id)
    if episode_memory not in existing:
        store.add_memory(
            session_id=session_id,
            kind=MemoryKind.EPISODE,
            status=EvidenceStatus.SUPPLIED,
            content=episode_memory,
            source="conversation user turn",
            confidence=0.9,
            importance=0.55,
        )
        existing.add(episode_memory)

    for candidate in _semantic_candidates(user_text):
        if candidate in existing:
            continue
        store.add_memory(
            session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            status=EvidenceStatus.SUPPLIED,
            content=candidate,
            source="Adrien conversation",
            confidence=0.8,
            importance=0.7,
        )
        existing.add(candidate)

    signal = segment(user_text)
    for question in signal.questions:
        dream = f"Unresolved exploration seed: {question}"
        if dream in existing:
            continue
        store.add_memory(
            session_id=session_id,
            kind=MemoryKind.DREAM,
            status=EvidenceStatus.SPECULATIVE,
            content=dream,
            source="heartbeat",
            confidence=0.3,
            importance=0.35,
        )
        existing.add(dream)
