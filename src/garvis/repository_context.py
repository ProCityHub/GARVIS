"""Read-only repository grounding for GARVIS responses."""

from __future__ import annotations

from pathlib import Path

GROUNDING_KEYWORDS = (
    "architecture",
    "code",
    "file",
    "implementation",
    "repository",
    "source",
    "test",
)

GROUNDING_FILES = (
    "src/garvis/assistant.py",
    "src/garvis/cli.py",
    "tests/garvis/test_assistant.py",
    "tests/garvis/test_cli.py",
    "tests/garvis/test_default_model.py",
)

MAX_FILE_CHARS = 4_000
MAX_CONTEXT_CHARS = 14_000


def should_ground_repository(message: str) -> bool:
    normalized = message.casefold()
    return any(keyword in normalized for keyword in GROUNDING_KEYWORDS)


def build_repository_context(repository_root: Path) -> str:
    root = repository_root.resolve()
    sections: list[str] = []

    for relative_name in GROUNDING_FILES:
        path = root / relative_name
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        excerpt = text[:MAX_FILE_CHARS]
        if len(text) > MAX_FILE_CHARS:
            excerpt += "\n[truncated]"

        sections.append(f"--- {relative_name} ---\n{excerpt}")

    return "\n\n".join(sections)[:MAX_CONTEXT_CHARS]


def ground_message(message: str, repository_root: Path) -> str:
    snapshot = build_repository_context(repository_root)

    if not snapshot:
        return message

    return (
        "[GARVIS read-only repository evidence]\n"
        "Use only the files below when making repository claims. "
        "Distinguish verified implementation from proposals or assumptions. "
        "You have no permission to modify files or execute commands.\n\n"
        f"{snapshot}\n\n"
        "[User request]\n"
        f"{message}"
    )
