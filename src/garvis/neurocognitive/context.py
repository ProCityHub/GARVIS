"""Working-context assembly for the GARVIS heartbeat."""

from __future__ import annotations

from .models import RecallResult


def _format_memory(result: RecallResult) -> str:
    memory = result.memory
    return (
        f"- [{memory.status.value}/{memory.kind.value}] "
        f"{memory.content} "
        f"(source={memory.source}; score={result.score:.2f})"
    )


def assemble_context(
    *,
    identity: tuple[RecallResult, ...],
    recalled: tuple[RecallResult, ...],
    contradictions: tuple[RecallResult, ...],
    intent: str,
    budget_chars: int = 14_000,
) -> str:
    sections: list[str] = [
        "[GARVIS neurocognitive working context]",
        "This is selective recall, not the complete archive.",
        f"Detected intent: {intent}",
        "",
        "0.0 Identity and authority:",
    ]

    sections.extend(_format_memory(item) for item in identity)
    sections.extend(["", "0.6 Relevant recalled memory:"])
    sections.extend(_format_memory(item) for item in recalled)

    if contradictions:
        sections.extend(["", "Contradictions and retractions:"])
        sections.extend(_format_memory(item) for item in contradictions)

    sections.extend(
        [
            "",
            "Evidence rule: distinguish verified, supplied, inferred, speculative, unknown, and retracted.",
            "Dream material may inspire hypotheses but is not evidence.",
            "External actions remain subject to the configured authority boundary.",
        ]
    )

    result = "\n".join(sections)
    return result[: max(2_000, budget_chars)]
