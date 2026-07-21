"""Selective recall and coherence scoring."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable

from .models import EvidenceStatus, MemoryRecord, RecallResult
from .segmentation import tokenize

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "garvis", "had", "has", "have", "he", "her", "here", "him",
    "his", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "she", "so", "system", "software", "architecture", "that", "the",
    "their", "them", "there", "they", "this", "to", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your", "adrien",
}

_STATUS_BONUS = {
    EvidenceStatus.VERIFIED: 0.30,
    EvidenceStatus.SUPPLIED: 0.20,
    EvidenceStatus.INFERRED: 0.08,
    EvidenceStatus.SPECULATIVE: -0.08,
    EvidenceStatus.UNKNOWN: -0.12,
    EvidenceStatus.RETRACTED: 0.25,
}


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 1 and token not in _STOPWORDS
    }


def _age_days(created_at: str) -> float:
    normalized = created_at.replace(" ", "T")
    try:
        created = datetime.fromisoformat(normalized)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(
        0.0,
        (datetime.now(timezone.utc) - created).total_seconds() / 86_400,
    )


def score_memory(query: str, memory: MemoryRecord) -> RecallResult:
    query_tokens = _meaningful_tokens(query)
    memory_tokens = _meaningful_tokens(memory.content)
    overlap = query_tokens & memory_tokens

    query_coverage = len(overlap) / max(1, len(query_tokens))
    memory_coverage = len(overlap) / max(1, len(memory_tokens))
    normalized_query = " ".join(query.casefold().split())
    normalized_memory = " ".join(memory.content.casefold().split())
    exact_phrase = (
        1.0
        if normalized_query
        and len(normalized_query) >= 8
        and normalized_query in normalized_memory
        else 0.0
    )
    recency = 1.0 / (1.0 + math.log1p(_age_days(memory.created_at)))
    context_cost = min(1.0, len(memory.content) / 6_000)

    if not overlap and not exact_phrase:
        score = 0.0
    else:
        score = (
            3.2 * query_coverage
            + 2.0 * memory_coverage
            + 1.4 * exact_phrase
            + 0.35 * memory.importance
            + 0.25 * memory.confidence
            + 0.12 * recency
            + _STATUS_BONUS[memory.status]
            - 0.18 * context_cost
        )

    reasons = (
        f"query_coverage={query_coverage:.3f}",
        f"memory_coverage={memory_coverage:.3f}",
        f"status={memory.status.value}",
        f"importance={memory.importance:.2f}",
        f"confidence={memory.confidence:.2f}",
    )
    return RecallResult(memory=memory, score=score, reasons=reasons)


def _dedup_key(memory: MemoryRecord) -> str:
    normalized = re.sub(r"\s+", " ", memory.content.casefold()).strip()
    return normalized[:1_000]


def recall(
    query: str,
    memories: Iterable[MemoryRecord],
    *,
    limit: int = 12,
) -> tuple[RecallResult, ...]:
    scored = [score_memory(query, memory) for memory in memories]
    scored.sort(key=lambda item: (item.score, item.memory.id), reverse=True)

    selected: list[RecallResult] = []
    seen: set[str] = set()
    for item in scored:
        if item.score < 0.72:
            continue
        key = _dedup_key(item.memory)
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= limit:
            break
    return tuple(selected)


def contradictions(
    recalled: Iterable[RecallResult],
) -> tuple[RecallResult, ...]:
    return tuple(
        result
        for result in recalled
        if result.memory.status is EvidenceStatus.RETRACTED
    )
