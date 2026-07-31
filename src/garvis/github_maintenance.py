"""Bounded Git and GitHub maintenance operations for GARVIS.

The adapter supports repository inspection and explicitly approved feature
branch maintenance. It blocks main-branch pushes, force pushes, destructive
Git operations, credential files, merges, deployments, and account changes.

Adrien D. Thomas retains final merge and external-action authority.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class MaintenancePolicyError(ValueError):
    """Raised when a requested maintenance operation violates policy."""


class MaintenanceCommandError(RuntimeError):
    """Raised when an approved command fails."""


@dataclass(frozen=True)
class CommandResult:
    """Immutable result of a maintenance command."""

    command: Tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RepositoryInspection:
    """Read-only summary of one Git repository."""

    repository: str
    branch: str
    status: str
    latest_commit: str
    remotes: str


_BLOCKED_BRANCHES = frozenset(
    {
        "main",
        "master",
        "production",
        "prod",
        "release",
    }
)

_BLOCKED_PATH_NAMES = frozenset(
    {
        ".env",
        "openai.env",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
        ".npmrc",
        ".pypirc",
    }
)

_BLOCKED_PATH_PARTS = frozenset(
    {
        ".git",
        ".ssh",
        ".gnupg",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
    }
)

_BLOCKED_ARGUMENTS = frozenset(
    {
        "--force",
        "-f",
        "--force-with-lease",
        "--mirror",
        "--delete",
        "--hard",
    }
)


def _safe_environment() -> Dict[str, str]:
    """Return a subprocess environment without common credential variables."""

    environment = dict(os.environ)

    blocked_names = {
        "OPENAI_API_KEY",
        "PROD_OPENAI_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }

    for name in blocked_names:
        environment.pop(name, None)

    return environment


def _validate_repository(repository: Path) -> Path:
    resolved = repository.expanduser().resolve()

    if not resolved.is_dir():
        raise MaintenancePolicyError(
            f"Repository directory does not exist: {resolved}"
        )

    git_marker = resolved / ".git"
    if not git_marker.exists():
        result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=_safe_environment(),
        )
        if result.returncode != 0:
            raise MaintenancePolicyError(
                f"Path is not a Git repository: {resolved}"
            )

    return resolved


def _validate_feature_branch(branch: str) -> str:
    cleaned = branch.strip()

    if not cleaned:
        raise MaintenancePolicyError("Branch name must not be empty")

    if cleaned.lower() in _BLOCKED_BRANCHES:
        raise MaintenancePolicyError(
            f"Protected branch cannot be modified: {cleaned}"
        )

    if cleaned.startswith("-"):
        raise MaintenancePolicyError("Branch name cannot begin with '-'")

    return cleaned


def _validate_relative_path(repository: Path, relative_path: str) -> str:
    candidate = Path(relative_path)

    if candidate.is_absolute():
        raise MaintenancePolicyError(
            f"Only repository-relative paths are allowed: {relative_path}"
        )

    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise MaintenancePolicyError(
            f"Unsafe repository path: {relative_path}"
        )

    if any(part in _BLOCKED_PATH_PARTS for part in candidate.parts):
        raise MaintenancePolicyError(
            f"Blocked repository path: {relative_path}"
        )

    if candidate.name.lower() in _BLOCKED_PATH_NAMES:
        raise MaintenancePolicyError(
            f"Credential or secret path is blocked: {relative_path}"
        )

    resolved = (repository / candidate).resolve()

    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise MaintenancePolicyError(
            f"Path escapes repository: {relative_path}"
        ) from exc

    return candidate.as_posix()


class GitHubMaintenanceAdapter:
    """Execute allowlisted Git and GitHub maintenance operations."""

    def __init__(self, timeout_seconds: int = 60) -> None:
        if timeout_seconds < 1:
            raise MaintenancePolicyError(
                "timeout_seconds must be greater than zero"
            )

        self.timeout_seconds = timeout_seconds

    def _run(
        self,
        command: Sequence[str],
        cwd: Optional[Path] = None,
    ) -> CommandResult:
        if not command:
            raise MaintenancePolicyError("Command must not be empty")

        if any(argument in _BLOCKED_ARGUMENTS for argument in command):
            raise MaintenancePolicyError(
                "Force, destructive, or deletion arguments are blocked"
            )

        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
            env=_safe_environment(),
        )

        result = CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

        if completed.returncode != 0:
            raise MaintenanceCommandError(
                "Command failed with exit code "
                f"{completed.returncode}: {' '.join(command)}\n"
                f"{completed.stderr.strip()}"
            )

        return result

    def inspect_repository(
        self,
        repository: Path,
    ) -> RepositoryInspection:
        """Inspect repository state without changing files or remote state."""

        repo = _validate_repository(repository)

        branch = self._run(
            ["git", "branch", "--show-current"],
            cwd=repo,
        ).stdout.strip()

        status = self._run(
            ["git", "status", "--short"],
            cwd=repo,
        ).stdout

        latest_commit = self._run(
            ["git", "log", "-1", "--oneline"],
            cwd=repo,
        ).stdout.strip()

        remotes = self._run(
            ["git", "remote", "-v"],
            cwd=repo,
        ).stdout

        return RepositoryInspection(
            repository=str(repo),
            branch=branch,
            status=status,
            latest_commit=latest_commit,
            remotes=remotes,
        )

    def list_pull_requests(
        self,
        github_repository: str,
        limit: int = 30,
    ) -> CommandResult:
        """List pull requests through the authenticated GitHub CLI."""

        if "/" not in github_repository:
            raise MaintenancePolicyError(
                "GitHub repository must use owner/name format"
            )

        if limit < 1 or limit > 100:
            raise MaintenancePolicyError(
                "Pull-request limit must be between 1 and 100"
            )

        return self._run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                github_repository,
                "--limit",
                str(limit),
                "--json",
                "number,title,state,headRefName,baseRefName,url",
            ]
        )

    def list_workflow_runs(
        self,
        github_repository: str,
        limit: int = 20,
    ) -> CommandResult:
        """List recent GitHub Actions runs without changing them."""

        if "/" not in github_repository:
            raise MaintenancePolicyError(
                "GitHub repository must use owner/name format"
            )

        if limit < 1 or limit > 100:
            raise MaintenancePolicyError(
                "Workflow-run limit must be between 1 and 100"
            )

        return self._run(
            [
                "gh",
                "run",
                "list",
                "--repo",
                github_repository,
                "--limit",
                str(limit),
                "--json",
                "databaseId,name,status,conclusion,headBranch,url",
            ]
        )

    def commit_feature_changes(
        self,
        repository: Path,
        branch: str,
        relative_paths: Sequence[str],
        message: str,
        approved: bool,
    ) -> CommandResult:
        """Stage exact files and create a commit on an approved feature branch."""

        if not approved:
            raise MaintenancePolicyError(
                "Explicit approval is required before committing"
            )

        repo = _validate_repository(repository)
        expected_branch = _validate_feature_branch(branch)

        current_branch = self._run(
            ["git", "branch", "--show-current"],
            cwd=repo,
        ).stdout.strip()

        if current_branch != expected_branch:
            raise MaintenancePolicyError(
                "Current branch does not match the approved feature branch"
            )

        if not relative_paths:
            raise MaintenancePolicyError(
                "At least one reviewed path is required"
            )

        safe_paths = [
            _validate_relative_path(repo, path)
            for path in relative_paths
        ]

        cleaned_message = message.strip()
        if not cleaned_message:
            raise MaintenancePolicyError(
                "Commit message must not be empty"
            )

        self._run(
            ["git", "add", "--", *safe_paths],
            cwd=repo,
        )

        return self._run(
            ["git", "commit", "-m", cleaned_message],
            cwd=repo,
        )

    def push_feature_branch(
        self,
        repository: Path,
        branch: str,
        approved: bool,
    ) -> CommandResult:
        """Push an explicitly approved feature branch without force."""

        if not approved:
            raise MaintenancePolicyError(
                "Explicit approval is required before pushing"
            )

        repo = _validate_repository(repository)
        safe_branch = _validate_feature_branch(branch)

        current_branch = self._run(
            ["git", "branch", "--show-current"],
            cwd=repo,
        ).stdout.strip()

        if current_branch != safe_branch:
            raise MaintenancePolicyError(
                "Current branch does not match the approved feature branch"
            )

        return self._run(
            ["git", "push", "-u", "origin", safe_branch],
            cwd=repo,
        )

    def open_pull_request(
        self,
        github_repository: str,
        head_branch: str,
        title: str,
        body: str,
        approved: bool,
        base_branch: str = "main",
    ) -> CommandResult:
        """Open an approved pull request without merging it."""

        if not approved:
            raise MaintenancePolicyError(
                "Explicit approval is required before opening a pull request"
            )

        safe_head = _validate_feature_branch(head_branch)

        if base_branch not in {"main", "master"}:
            raise MaintenancePolicyError(
                "Pull-request base must be main or master"
            )

        if not title.strip():
            raise MaintenancePolicyError(
                "Pull-request title must not be empty"
            )

        if "/" not in github_repository:
            raise MaintenancePolicyError(
                "GitHub repository must use owner/name format"
            )

        return self._run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                github_repository,
                "--base",
                base_branch,
                "--head",
                safe_head,
                "--title",
                title.strip(),
                "--body",
                body,
            ]
        )

    def merge_pull_request(self, *_args: object, **_kwargs: object) -> None:
        """Block merges; Adrien D. Thomas retains final merge authority."""

        raise MaintenancePolicyError(
            "Pull-request merging is not permitted through this adapter"
        )
