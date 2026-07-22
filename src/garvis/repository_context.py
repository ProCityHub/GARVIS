"""Read-only repository grounding for GARVIS responses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

GROUNDING_KEYWORDS = (
    "architecture",
    "branch",
    "bug",
    "capability",
    "class",
    "code",
    "commit",
    "file",
    "function",
    "github",
    "implementation",
    "module",
    "pull request",
    "repository",
    "runtime",
    "source",
    "test",
)

# Preserve the legacy remote-runtime defaults.
GROUNDING_FILES = (
    "src/garvis/assistant.py",
    "src/garvis/cli.py",
    "tests/garvis/test_assistant.py",
    "tests/garvis/test_cli.py",
    "tests/garvis/test_default_model.py",
)

# Safe starting points for the local default runtime.
LOCAL_GROUNDING_FILES = (
    "src/garvis/repository_context.py",
    "src/garvis/local_language_runtime.py",
    "src/garvis/capability_runtime.py",
    "src/garvis/capability_broker.py",
    "src/garvis/internet_research.py",
    "src/garvis/memory_lifecycle.py",
    "src/garvis/phone_capabilities.py",
    "src/garvis/cli.py",
)

DISCOVERY_ROOTS = (
    "src/garvis",
    "tests/garvis",
    "scripts",
    "config",
    "docs",
)

SAFE_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

SENSITIVE_NAMES = {
    ".env",
    ".git",
    ".secrets",
    "credentials",
    "id_ed25519",
    "id_rsa",
    "private_key",
    "secrets",
    "wallet",
}

SENSITIVE_FRAGMENTS = (
    "app_password",
    "credential",
    "mnemonic",
    "password",
    "private-key",
    "private_key",
    "secret",
    "seed_phrase",
)

MAX_FILE_CHARS = 4_000
MAX_CONTEXT_CHARS = 14_000
LOCAL_MAX_FILE_CHARS = 700
LOCAL_MAX_CONTEXT_CHARS = 5_000
LOCAL_MAX_FILES = 8

_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}


def should_ground_repository(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return any(keyword in normalized for keyword in GROUNDING_KEYWORDS)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def _is_sensitive(relative_name: str) -> bool:
    lowered = relative_name.casefold()
    parts = {part.casefold() for part in Path(relative_name).parts}
    name = Path(relative_name).name.casefold()
    return (
        bool(parts & SENSITIVE_NAMES)
        or name in SENSITIVE_NAMES
        or any(fragment in lowered for fragment in SENSITIVE_FRAGMENTS)
    )


def _safe_relative_file(root: Path, relative_name: str) -> Path | None:
    if _is_sensitive(relative_name):
        return None
    path = root / relative_name
    if path.is_symlink() or not path.is_file():
        return None
    if path.suffix.casefold() not in SAFE_TEXT_SUFFIXES:
        return None
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return None
    return path


def build_repository_context(
    repository_root: Path,
    *,
    file_names: Sequence[str] = GROUNDING_FILES,
    max_file_chars: int = MAX_FILE_CHARS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    root = repository_root.resolve()
    if max_file_chars < 1 or max_context_chars < 1:
        return ""

    sections: list[str] = []
    used = 0
    for relative_name in file_names:
        path = _safe_relative_file(root, relative_name)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        excerpt = text[:max_file_chars]
        if len(text) > max_file_chars:
            excerpt += "\n[truncated]"
        section = f"--- {relative_name} ---\n{excerpt}"

        separator = 2 if sections else 0
        remaining = max_context_chars - used - separator
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining]
        sections.append(section)
        used += separator + len(section)
        if used >= max_context_chars:
            break

    return "\n\n".join(sections)[:max_context_chars]


def _discover_candidates(root: Path) -> tuple[str, ...]:
    candidates: set[str] = set(LOCAL_GROUNDING_FILES)
    for relative_root in DISCOVERY_ROOTS:
        directory = root / relative_root
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.rglob("*"):
            if len(candidates) >= 500:
                break
            if path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if _is_sensitive(relative):
                continue
            if path.suffix.casefold() not in SAFE_TEXT_SUFFIXES:
                continue
            candidates.add(relative)
    return tuple(sorted(candidates))


def _score_candidate(root: Path, relative_name: str, query_tokens: set[str]) -> int:
    path_tokens = _tokens(relative_name.replace("/", " "))
    score = 4 * len(query_tokens & path_tokens)
    path = _safe_relative_file(root, relative_name)
    if path is None:
        return -1
    try:
        excerpt = path.read_text(encoding="utf-8")[:5_000]
    except (OSError, UnicodeError):
        return -1
    score += len(query_tokens & _tokens(excerpt))
    if relative_name in LOCAL_GROUNDING_FILES:
        score += 1
    return score


def select_repository_files(
    repository_root: Path,
    query: str,
    *,
    max_files: int = LOCAL_MAX_FILES,
) -> tuple[str, ...]:
    root = repository_root.resolve()
    query_tokens = _tokens(query)
    ranked = [
        (_score_candidate(root, relative_name, query_tokens), relative_name)
        for relative_name in _discover_candidates(root)
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [name for score, name in ranked if score > 0][: max(1, max_files)]
    if not selected:
        selected = list(LOCAL_GROUNDING_FILES[: max(1, max_files)])
    return tuple(selected)


def build_query_repository_context(
    repository_root: Path,
    query: str,
    *,
    max_files: int = LOCAL_MAX_FILES,
    max_file_chars: int = LOCAL_MAX_FILE_CHARS,
    max_context_chars: int = LOCAL_MAX_CONTEXT_CHARS,
) -> str:
    selected = select_repository_files(repository_root, query, max_files=max_files)
    return build_repository_context(
        repository_root,
        file_names=selected,
        max_file_chars=max_file_chars,
        max_context_chars=max_context_chars,
    )


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
