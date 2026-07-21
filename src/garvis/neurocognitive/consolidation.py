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


def consolidate_turn(
    store: NeuroStore,
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    store.add_episode(session_id, user_text, assistant_text)

    episode_summary = (
        f"Adrien: {user_text[:1_500]}\n"
        f"GARVIS: {assistant_text[:1_500]}"
    )
    store.add_memory(
        session_id=session_id,
        kind=MemoryKind.EPISODE,
        status=EvidenceStatus.SUPPLIED,
        content=episode_summary,
        source="conversation",
        confidence=0.9,
        importance=0.55,
    )

    for candidate in _semantic_candidates(user_text):
        store.add_memory(
            session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            status=EvidenceStatus.SUPPLIED,
            content=candidate,
            source="Adrien conversation",
            confidence=0.8,
            importance=0.7,
        )

    signal = segment(user_text)
    for question in signal.questions:
        store.add_memory(
            session_id=session_id,
            kind=MemoryKind.DREAM,
            status=EvidenceStatus.SPECULATIVE,
            content=f"Unresolved exploration seed: {question}",
            source="heartbeat",
            confidence=0.3,
            importance=0.35,
        )
