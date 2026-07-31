from datetime import datetime, timezone
from pathlib import Path

import pytest

from garvis.local_file_access import (
    LocalAccessError,
    LocalAccessRequest,
    LocalAccessState,
    LocalFileAccessStore,
    appears_to_require_local_access,
    execute_local_access,
    parse_local_access_request,
)


def test_parser_requires_explicit_file_operation(tmp_path: Path) -> None:
    assert not appears_to_require_local_access("Explain the repository architecture")
    message = f'Read file "{tmp_path / "note.txt"}"'
    assert appears_to_require_local_access(message)
    target, operation, query = parse_local_access_request(message, tmp_path)
    assert target == str(tmp_path / "note.txt")
    assert operation == "read"
    assert query == ""


def test_one_task_approval_and_read(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("verified local evidence", encoding="utf-8")
    store = LocalFileAccessStore(tmp_path / "access.db")
    request = store.create(
        f'Read file "{note}"',
        str(note),
        "read",
        "",
    )
    assert "Approve? [Y/N]" in request.render()
    resolution = store.resolve("yes")
    assert resolution is not None and resolution.approved
    report = execute_local_access(
        resolution.request,
        tmp_path,
        roots=(tmp_path.resolve(),),
    )
    assert "verified local evidence" in report.content
    assert note.read_text(encoding="utf-8") == "verified local evidence"
    store.close()


def test_secret_paths_are_permanently_excluded(tmp_path: Path) -> None:
    secret = tmp_path / ".secrets" / "credential.txt"
    secret.parent.mkdir()
    secret.write_text("never read", encoding="utf-8")
    now = datetime.now(timezone.utc)
    request = LocalAccessRequest(
        "id",
        "default",
        "read secret",
        str(secret),
        "read",
        "",
        LocalAccessState.APPROVED,
        now,
        now,
    )
    with pytest.raises(LocalAccessError, match="secret-exclusion"):
        execute_local_access(request, tmp_path, roots=(tmp_path.resolve(),))


def test_outside_allowed_roots_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    now = datetime.now(timezone.utc)
    request = LocalAccessRequest(
        "id",
        "default",
        "read outside",
        str(outside),
        "read",
        "",
        LocalAccessState.APPROVED,
        now,
        now,
    )
    with pytest.raises(LocalAccessError, match="outside the approved"):
        execute_local_access(request, tmp_path, roots=(tmp_path.resolve(),))


def test_directory_list_is_top_level_and_does_not_open_files(tmp_path: Path) -> None:
    visible = tmp_path / "visible.txt"
    visible.write_text("content must not be returned", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("deep content", encoding="utf-8")
    secret = tmp_path / ".secrets"
    secret.mkdir()
    (secret / "hidden.txt").write_text("hidden", encoding="utf-8")

    now = datetime.now(timezone.utc)
    request = LocalAccessRequest(
        "id",
        "default",
        "list directory",
        str(tmp_path),
        "list",
        "",
        LocalAccessState.APPROVED,
        now,
        now,
    )
    report = execute_local_access(request, tmp_path, roots=(tmp_path.resolve(),))

    assert "visible.txt" in report.content
    assert "nested/" in report.content
    assert "deep.txt" not in report.content
    assert "content must not be returned" not in report.content
    assert ".secrets" not in report.content
