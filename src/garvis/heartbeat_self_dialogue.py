"""Bounded internal self-dialogue for the GARVIS heartbeat.

This module observes local software state and produces deterministic
Observer/Skeptic/Planner dialogue. Dialogue is interpretation, not evidence,
and never grants execution authority.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class SystemSnapshot:
    repository_root: str
    repository_available: bool
    branch: str
    head_sha: str
    dirty_paths: Tuple[str, ...]
    pending_protected_actions: int
    observed_at: float

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "repository_root": self.repository_root,
            "repository_available": self.repository_available,
            "branch": self.branch,
            "head_sha": self.head_sha,
            "dirty_paths": list(self.dirty_paths),
            "pending_protected_actions": self.pending_protected_actions,
            "observed_at": self.observed_at,
        }


def _git(repository_root: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
        env=environment,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def observe_system(
    repository_root: Path,
    pending_protected_actions: int,
) -> SystemSnapshot:
    root = Path(repository_root).expanduser().resolve()
    top = _git(root, "rev-parse", "--show-toplevel")
    available = bool(top)

    branch = _git(root, "branch", "--show-current") if available else ""
    head = _git(root, "rev-parse", "HEAD") if available else ""
    status = (
        _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
        if available
        else ""
    )

    dirty = tuple(
        line[3:].strip()
        for line in status.splitlines()
        if len(line) >= 4 and line[3:].strip()
    )

    return SystemSnapshot(
        repository_root=str(root),
        repository_available=available,
        branch=branch,
        head_sha=head,
        dirty_paths=dirty,
        pending_protected_actions=int(pending_protected_actions),
        observed_at=time.time(),
    )


def build_internal_dialogue(snapshot: SystemSnapshot) -> Mapping[str, Any]:
    concerns = []

    if not snapshot.repository_available:
        concerns.append("repository observation is unavailable")
    if snapshot.repository_available and not snapshot.branch:
        concerns.append("repository HEAD is detached")
    if snapshot.branch and snapshot.branch != "main":
        concerns.append(
            "runtime checkout is not on main; preserve exact ancestry before promotion"
        )
    if snapshot.dirty_paths:
        concerns.append(
            f"working tree contains {len(snapshot.dirty_paths)} changed path(s)"
        )
    if snapshot.pending_protected_actions:
        concerns.append(
            f"{snapshot.pending_protected_actions} protected action(s) are queued"
        )

    short_head = snapshot.head_sha[:12] if snapshot.head_sha else "unknown"
    observer = (
        f"Repository={snapshot.repository_available}; "
        f"branch={snapshot.branch or 'detached/unknown'}; "
        f"HEAD={short_head}; "
        f"dirty_paths={len(snapshot.dirty_paths)}; "
        f"pending_protected_actions={snapshot.pending_protected_actions}."
    )

    if concerns:
        skeptic = "I see a possible contradiction: " + "; ".join(concerns) + "."
    else:
        skeptic = (
            "I do not see a local repository contradiction in this observation. "
            "That does not prove the whole system is healthy."
        )

    if snapshot.dirty_paths:
        repair_candidate = "preserve_then_diagnose_worktree"
        planner = (
            "Preserve the changed paths, inspect their provenance, test a repair in "
            "isolation, and do not discard work."
        )
    elif snapshot.pending_protected_actions:
        repair_candidate = "review_queued_protected_actions"
        planner = (
            "Keep the heartbeat running while protected actions remain queued for "
            "their separate authorization path."
        )
    elif not snapshot.repository_available:
        repair_candidate = "restore_repository_observation"
        planner = (
            "Restore read-only repository observation before claiming repository health."
        )
    elif snapshot.branch and snapshot.branch != "main":
        repair_candidate = "verify_branch_promotion_state"
        planner = (
            "Compare this exact HEAD with canonical main before treating the checkout "
            "as promoted."
        )
    else:
        repair_candidate = "no_repair_needed"
        planner = (
            "Continue observing. Make no repository mutation without a concrete "
            "contradiction and verified repair candidate."
        )

    return {
        "status": "INTERNAL_DIALOGUE_NOT_EVIDENCE",
        "observer": observer,
        "skeptic": skeptic,
        "planner": planner,
        "repair_candidate": repair_candidate,
        "concerns": concerns,
    }
