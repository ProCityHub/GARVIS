from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pytest

from garvis.github_maintenance import (
    CommandResult,
    GitHubMaintenanceAdapter,
    MaintenancePolicyError,
)


class RecordingAdapter(GitHubMaintenanceAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[tuple[str, ...], Optional[Path]]] = []
        self.branch = "feature/test-maintenance"

    def _run(
        self,
        command: Sequence[str],
        cwd: Optional[Path] = None,
    ) -> CommandResult:
        recorded = tuple(command)
        self.commands.append((recorded, cwd))

        if recorded == ("git", "branch", "--show-current"):
            stdout = self.branch + "\n"
        elif recorded == ("git", "status", "--short"):
            stdout = ""
        elif recorded == ("git", "log", "-1", "--oneline"):
            stdout = "abc1234 Test commit\n"
        elif recorded == ("git", "remote", "-v"):
            stdout = "origin git@github.com:ProCityHub/GARVIS.git\n"
        else:
            stdout = "ok\n"

        return CommandResult(
            command=recorded,
            returncode=0,
            stdout=stdout,
            stderr="",
        )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_repository_inspection_is_read_only(repository: Path) -> None:
    adapter = RecordingAdapter()

    result = adapter.inspect_repository(repository)

    assert result.branch == "feature/test-maintenance"
    assert result.latest_commit == "abc1234 Test commit"
    assert ("git", "status", "--short") in [command for command, _cwd in adapter.commands]
    assert not any(
        command[:2] in {("git", "add"), ("git", "commit")} for command, _cwd in adapter.commands
    )


def test_pull_request_listing_uses_read_only_command() -> None:
    adapter = RecordingAdapter()

    adapter.list_pull_requests("ProCityHub/GARVIS")

    command = adapter.commands[-1][0]
    assert command[:3] == ("gh", "pr", "list")
    assert "create" not in command


def test_workflow_listing_uses_read_only_command() -> None:
    adapter = RecordingAdapter()

    adapter.list_workflow_runs("ProCityHub/GARVIS")

    command = adapter.commands[-1][0]
    assert command[:3] == ("gh", "run", "list")


def test_commit_requires_explicit_approval(repository: Path) -> None:
    adapter = RecordingAdapter()

    with pytest.raises(
        MaintenancePolicyError,
        match="Explicit approval",
    ):
        adapter.commit_feature_changes(
            repository=repository,
            branch="feature/test-maintenance",
            relative_paths=["src/garvis/example.py"],
            message="Test change",
            approved=False,
        )


def test_commit_stages_only_reviewed_paths(repository: Path) -> None:
    adapter = RecordingAdapter()

    adapter.commit_feature_changes(
        repository=repository,
        branch="feature/test-maintenance",
        relative_paths=[
            "src/garvis/example.py",
            "tests/garvis/test_example.py",
        ],
        message="Test reviewed change",
        approved=True,
    )

    commands = [command for command, _cwd in adapter.commands]

    assert (
        "git",
        "add",
        "--",
        "src/garvis/example.py",
        "tests/garvis/test_example.py",
    ) in commands

    assert (
        "git",
        "commit",
        "-m",
        "Test reviewed change",
    ) in commands


@pytest.mark.parametrize(
    "blocked_path",
    [
        ".env",
        "openai.env",
        "credentials.json",
        ".ssh/id_rsa",
        "../outside.txt",
        ".git/config",
    ],
)
def test_commit_blocks_secret_or_unsafe_paths(
    repository: Path,
    blocked_path: str,
) -> None:
    adapter = RecordingAdapter()

    with pytest.raises(MaintenancePolicyError):
        adapter.commit_feature_changes(
            repository=repository,
            branch="feature/test-maintenance",
            relative_paths=[blocked_path],
            message="Unsafe change",
            approved=True,
        )


@pytest.mark.parametrize(
    "branch",
    ["main", "master", "production", "prod", "release"],
)
def test_push_blocks_protected_branches(
    repository: Path,
    branch: str,
) -> None:
    adapter = RecordingAdapter()
    adapter.branch = branch

    with pytest.raises(
        MaintenancePolicyError,
        match="Protected branch",
    ):
        adapter.push_feature_branch(
            repository=repository,
            branch=branch,
            approved=True,
        )


def test_push_requires_explicit_approval(repository: Path) -> None:
    adapter = RecordingAdapter()

    with pytest.raises(
        MaintenancePolicyError,
        match="Explicit approval",
    ):
        adapter.push_feature_branch(
            repository=repository,
            branch="feature/test-maintenance",
            approved=False,
        )


def test_push_feature_branch_never_uses_force(repository: Path) -> None:
    adapter = RecordingAdapter()

    adapter.push_feature_branch(
        repository=repository,
        branch="feature/test-maintenance",
        approved=True,
    )

    command = adapter.commands[-1][0]
    assert command == (
        "git",
        "push",
        "-u",
        "origin",
        "feature/test-maintenance",
    )
    assert "--force" not in command
    assert "-f" not in command


def test_open_pull_request_requires_approval() -> None:
    adapter = RecordingAdapter()

    with pytest.raises(
        MaintenancePolicyError,
        match="Explicit approval",
    ):
        adapter.open_pull_request(
            github_repository="ProCityHub/GARVIS",
            head_branch="feature/test-maintenance",
            title="Test pull request",
            body="Test body",
            approved=False,
        )


def test_open_pull_request_does_not_merge() -> None:
    adapter = RecordingAdapter()

    adapter.open_pull_request(
        github_repository="ProCityHub/GARVIS",
        head_branch="feature/test-maintenance",
        title="Test pull request",
        body="Test body",
        approved=True,
    )

    command = adapter.commands[-1][0]
    assert command[:3] == ("gh", "pr", "create")
    assert "merge" not in command


def test_merge_is_always_blocked() -> None:
    adapter = RecordingAdapter()

    with pytest.raises(
        MaintenancePolicyError,
        match="merging is not permitted",
    ):
        adapter.merge_pull_request(27)
