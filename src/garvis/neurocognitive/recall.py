"""Selective recall and coherence scoring."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from .models import EvidenceStatus, MemoryRecord, RecallResult
from .segmentation import tokenize

_STATUS_WEIGHT = {
    EvidenceStatus.VERIFIED: 1.0,
    EvidenceStatus.SUPPLIED: 0.85,
    EvidenceStatus.INFERRED: 0.65,
    EvidenceStatus.SPECULATIVE: 0.35,
    EvidenceStatus.UNKNOWN: 0.15,
    EvidenceStatus.RETRACTED: -1.0,
}


def _age_days(created_at: str) -> float:
    normalized = created_at.replace(" ", "T")
    try:
        created = datetime.fromisoformat(normalized)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86_400)


def score_memory(query: str, memory: MemoryRecord) -> RecallResult:
    query_tokens = set(tokenize(query))
    memory_tokens = set(tokenize(memory.content))
    overlap = len(query_tokens & memory_tokens)
    union = max(1, len(query_tokens | memory_tokens))
    semantic_overlap = overlap / union
    exact_phrase = 1.0 if query.casefold() in memory.content.casefold() else 0.0
    status_weight = _STATUS_WEIGHT[memory.status]
    recency = 1.0 / (1.0 + math.log1p(_age_days(memory.created_at)))
    context_cost = min(1.0, len(memory.content) / 8_000)

    score = (
        4.0 * semantic_overlap
        + 1.5 * exact_phrase
        + 1.4 * memory.importance
        + 1.0 * memory.confidence
        + 0.6 * recency
        + 1.2 * status_weight
        - 0.5 * context_cost
    )

    reasons = (
        f"overlap={semantic_overlap:.3f}",
        f"status={memory.status.value}",
        f"importance={memory.importance:.2f}",
        f"confidence={memory.confidence:.2f}",
        f"recency={recency:.2f}",
    )
    return RecallResult(memory=memory, score=score, reasons=reasons)


def recall(
    query: str,
    memories: Iterable[MemoryRecord],
    *,
    limit: int = 12,
) -> tuple[RecallResult, ...]:
    scored = [score_memory(query, memory) for memory in memories]
    scored.sort(key=lambda item: (item.score, item.memory.id), reverse=True)
    return tuple(item for item in scored if item.score > 0.8)[:limit]


def contradictions(
    recalled: Iterable[RecallResult],
) -> tuple[RecallResult, ...]:
    return tuple(
        result
        for result in recalled
        if result.memory.status is EvidenceStatus.RETRACTED
    )
