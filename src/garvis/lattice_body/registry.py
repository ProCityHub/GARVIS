"""Deny-by-default tool registry.

Irreversible tools are refused even if a caller asks for them by name.
Merge, delete, and force-push also pass through the existing GitHub
maintenance blocks, which raise before any git subprocess runs.
Network research is handed to the capability-broker phrase check and then
stopped: this dry-run performs no network I/O.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from garvis.capability_broker import (
    ApprovalStore,
    extract_research_query,
    has_explicit_network_authorization,
)
from garvis.github_maintenance import (
    GitHubMaintenanceAdapter,
    MaintenancePolicyError,
)
from garvis.local_file_access import (
    LocalAccessError,
    LocalFileAccessStore,
    execute_local_access,
)
from .redact import redact_text

Privilege = str

READ = "read"
VERIFY = "verify"
DRAFT = "draft"
NETWORK = "network"
IRREVERSIBLE = "irreversible"

_ALWAYS_REFUSED = {
    "merge": "merge",
    "merge_pull_request": "merge",
    "delete": "delete",
    "close": "close",
    "force-push": "force-push",
    "force_push": "force-push",
    "forcepush": "force-push",
    "send": "send",
    "outbound_send": "send",
    "outbound-send": "send",
    "deploy": "deploy",
    "ruleset_change": "ruleset",
}


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    refused: bool
    privilege: str
    reason: str
    output: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


Handler = Callable[..., str]


class ToolRegistry:
    """Callable tools must be registered with a privilege class."""

    def __init__(self, root: Path):
        self.root = root
        self._tools: dict[str, tuple[str, Handler]] = {}
        self.network_calls = 0

    def register(self, name: str, privilege: str, handler: Handler) -> None:
        if name in _ALWAYS_REFUSED or privilege == IRREVERSIBLE:
            raise ValueError(
                f"refused: {name} is irreversible and cannot be registered"
            )
        if privilege not in {READ, VERIFY, DRAFT, NETWORK}:
            raise ValueError(f"unknown privilege class: {privilege}")
        self._tools[name] = (privilege, handler)

    def invoke(self, name: str, /, **kwargs: object) -> ToolResult:
        blocked = _ALWAYS_REFUSED.get(name)
        if blocked is not None:
            return ToolResult(
                name=name,
                ok=False,
                refused=True,
                privilege=IRREVERSIBLE,
                reason="refused: " + _maintenance_reason(blocked),
                output="",
            )
        registered = self._tools.get(name)
        if registered is None:
            return ToolResult(
                name=name,
                ok=False,
                refused=True,
                privilege="denied",
                reason="refused: deny-by-default; tool is not registered",
                output="",
            )
        privilege, handler = registered
        try:
            if privilege == NETWORK:
                return self._network(name, str(kwargs.get("message") or ""))
            output = handler(**kwargs)
        except (OSError, ValueError, LocalAccessError, sqlite3.Error) as exc:
            return ToolResult(
                name=name,
                ok=False,
                refused=privilege == NETWORK,
                privilege=privilege,
                reason=redact_text(str(exc)) or "tool failed",
                output="",
            )
        return ToolResult(
            name=name,
            ok=True,
            refused=False,
            privilege=privilege,
            reason="proposal" if privilege == DRAFT else "completed",
            output=redact_text(output),
        )

    def _network(self, name: str, message: str) -> ToolResult:
        """Record the broker decision and do not call internet research."""

        store = ApprovalStore(db_path=self.root / ".lattice-capability-broker.sqlite")
        try:
            store.create(
                message,
                extract_research_query(message),
                session_id="lattice-body",
            )
            authorized = has_explicit_network_authorization(message)
            if authorized:
                store.resolve("approve", session_id="lattice-body")
                reason = (
                    "refused: capability broker recorded approval; "
                    "dry-run does not perform network research"
                )
            else:
                reason = (
                    "refused: network research requires capability-broker "
                    "approval; no request was made"
                )
        finally:
            store.close()
        return ToolResult(
            name=name,
            ok=False,
            refused=True,
            privilege=NETWORK,
            reason=reason,
            output="",
        )


def _maintenance_reason(kind: str) -> str:
    """Reuse maintenance blocks that raise before a subprocess starts."""

    adapter = GitHubMaintenanceAdapter()
    try:
        if kind == "merge":
            adapter.merge_pull_request()
        elif kind == "force-push":
            adapter._run(["git", "push", "--force"])
        elif kind == "delete":
            adapter._run(["git", "push", "--delete", "blocked-ref"])
        else:
            return (
                f"Law 0 blocks {kind}. The body proposes only. "
                "No message was sent and no issue was closed."
            )
    except MaintenancePolicyError as exc:
        return (
            f"Law 0 blocks {kind}. The body proposes only. "
            f"Existing maintenance gate: {exc}"
        )
    return f"Law 0 blocks {kind}. The body proposes only."


def _contained(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("path must stay inside the cycle workspace")
    resolved = (root / candidate).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def build_tool_registry(root: Path) -> ToolRegistry:
    """Register the dry-run read, verify, and draft tools."""

    registry = ToolRegistry(root)

    def file_read(path: str = "", **_kwargs: object) -> str:
        target = _contained(root, path)
        session_id = "lattice-body-" + uuid.uuid4().hex
        store = LocalFileAccessStore(db_path=root / ".lattice-local-access.sqlite")
        try:
            store.create(
                f'read "{target}"',
                str(target),
                "read",
                "",
                session_id=session_id,
            )
            resolution = store.resolve("approve", session_id=session_id)
            if resolution is None or not resolution.approved:
                raise LocalAccessError("local file access was not approved")
            report = execute_local_access(resolution.request, root)
        finally:
            store.close()
        return report.content

    def repo_status(**_kwargs: object) -> str:
        return f"local workspace {root.name}; no remote call"

    def evidence_verify(**_kwargs: object) -> str:
        return "verification runs through the evidence verifier"

    def draft_note(text: str = "", **_kwargs: object) -> str:
        return "queued proposal: " + text

    registry.register("file_read", READ, file_read)
    registry.register("repo_status", READ, repo_status)
    registry.register("evidence_verify", VERIFY, evidence_verify)
    registry.register("draft_note", DRAFT, draft_note)
    registry.register("network_research", NETWORK, lambda **_kwargs: "")
    return registry
