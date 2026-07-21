"""Typed records used by the GARVIS neurocognitive engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    SUPPLIED = "supplied"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"
    UNKNOWN = "unknown"
    RETRACTED = "retracted"


class MemoryKind(str, Enum):
    IDENTITY = "identity"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    EVIDENCE = "evidence"
    TASK = "task"
    DREAM = "dream"
    FEEDBACK = "feedback"


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    created_at: str
    session_id: str
    kind: MemoryKind
    status: EvidenceStatus
    content: str
    source: str
    confidence: float
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallResult:
    memory: MemoryRecord
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CycleContext:
    session_id: str
    raw_input: str
    segmented_sentences: tuple[str, ...]
    intent: str
    recalled: tuple[RecallResult, ...]
    contradictions: tuple[RecallResult, ...]
    model_context: str
