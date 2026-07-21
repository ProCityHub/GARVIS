"""Deterministic signal segmentation for text input."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@'/-]*")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class SegmentedSignal:
    sentences: tuple[str, ...]
    tokens: tuple[str, ...]
    intent: str
    questions: tuple[str, ...]


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(text))


def segment(text: str) -> SegmentedSignal:
    sentences = tuple(
        part.strip()
        for part in _SENTENCE_PATTERN.split(text.strip())
        if part.strip()
    )
    tokens = tokenize(text)
    lowered = text.strip().casefold()

    if lowered.endswith("?") or lowered.startswith(
        ("what ", "why ", "how ", "when ", "where ", "who ", "can ", "could ")
    ):
        intent = "question"
    elif lowered.startswith(
        (
            "send ",
            "email ",
            "create ",
            "delete ",
            "buy ",
            "sell ",
            "trade ",
            "transfer ",
            "submit ",
            "publish ",
        )
    ):
        intent = "external_action_request"
    elif lowered.startswith(("remember ", "record ", "save ", "note ")):
        intent = "memory_instruction"
    elif lowered.startswith(("dream ", "imagine ", "speculate ", "hypothesize ")):
        intent = "dream_or_speculation"
    else:
        intent = "conversation"

    questions = tuple(sentence for sentence in sentences if sentence.endswith("?"))
    return SegmentedSignal(
        sentences=sentences,
        tokens=tokens,
        intent=intent,
        questions=questions,
    )
