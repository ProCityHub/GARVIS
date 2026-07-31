"""THANOS MODE standing authorization for GARVIS self-maintenance.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

THANOS MODE is a persistent, revocable, repository-scoped standing
authorization. While it is enabled, GARVIS may run its own maintenance
lifecycle -- research, diagnose, patch, test, commit, push, open and repair
pull requests -- without emitting a per-stage approval prompt.

An active authorization permits autonomous squash-merge of ordinary GARVIS
changes once every merge precondition in ``MergePreconditions`` holds.

Governance paths (``PROTECTED_PATHS``) are the single exception. A cycle may
freely research, patch, test and open pull requests against them, but it
cannot self-certify the merge, because the validation profile for those
files -- the test suite, the CI workflows, this module -- is exactly what
such a change edits. Enforcement lives in GitHub branch protection and
CODEOWNERS, outside the runtime GARVIS can rewrite.

Python 3.9 compatible: no ``datetime.UTC``, no ``slots=True`` dataclasses,
no 3.10-only union syntax at runtime.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from garvis.stage_gate import (
    canonical_json,
    new_identifier,
    parse_utc_iso,
    sha256_payload,
    utc_now_iso,
)

__all__ = [
    "AUTHORIZATION_VERSION",
    "DEFAULT_ALLOWED_ACTIONS",
    "OWNER",
    "PROTECTED_PATHS",
    "REPOSITORY",
    "RUNTIME_SCOPE",
    "ThanosAction",
    "ThanosAuthorization",
    "ThanosAuthorizationStore",
    "ThanosError",
    "ThanosNotEnabledError",
    "ThanosPausedError",
    "ThanosRevokedError",
    "ThanosScopeError",
    "ThanosTamperError",
    "MergeDecision",
    "MergePreconditions",
    "create_authorization",
    "evaluate_merge_gate",
    "is_protected_path",
    "pause_authorization",
    "permits",
    "render_status",
    "resume_authorization",
    "revoke_authorization",
]


OWNER = "Adrien D. Thomas"
REPOSITORY = "ProCityHub/GARVIS"
RUNTIME_SCOPE = "garvis-runtime"
AUTHORITY = "autonomous-self-upgrade"
AUTHORIZATION_VERSION = 1
GENESIS_HASH = "0" * 64


class ThanosError(RuntimeError):
    """Base error for THANOS MODE."""


class ThanosNotEnabledError(ThanosError):
    """Raised when protected work is attempted without an enabled grant."""


class ThanosPausedError(ThanosError):
    """Raised when the standing authorization is paused."""


class ThanosRevokedError(ThanosError):
    """Raised when the standing authorization has been revoked."""


class ThanosScopeError(ThanosError):
    """Raised when an action falls outside the declared scope."""


class ThanosTamperError(ThanosError):
    """Raised when the persisted authorization chain fails verification."""


class ThanosAction(str, Enum):
    """Actions a THANOS cycle may perform without a per-stage prompt."""

    RESEARCH = "research"
    INSPECT = "inspect"
    PLAN = "plan"
    EDIT = "edit"
    CREATE = "create"
    REFACTOR = "refactor"
    DEPENDENCY_UPDATE = "dependency-update"
    TEST = "test"
    SECURITY_REVIEW = "security-review"
    PACKAGE = "package"
    COMMIT = "commit"
    PUSH = "push"
    CREATE_PULL_REQUEST = "create-pull-request"
    UPDATE_PULL_REQUEST = "update-pull-request"
    MONITOR_CI = "monitor-ci"
    REPAIR_CI = "repair-ci"
    REQUEST_MERGE = "request-merge"
    RESTART = "restart"
    ROLLBACK = "rollback"
    CONTINUE_UPGRADING = "continue-upgrading"

    MERGE = "merge"


#: The full standing grant. MERGE is included: an active authorization
#: permits autonomous squash-merge of ordinary GARVIS changes.
DEFAULT_ALLOWED_ACTIONS = tuple(ThanosAction)

#: Paths whose modification must not be merged by an autonomous cycle.
#: The authoritative enforcement is GitHub branch protection + CODEOWNERS;
#: this tuple is defense in depth on the local side.
PROTECTED_PATHS = (
    ".github/CODEOWNERS",
    ".github/workflows/",
    "src/garvis/thanos_mode.py",
    "src/garvis/stage_gate.py",
    "src/garvis/stage_gate_store.py",
    "src/garvis/github_maintenance.py",
    "src/garvis/upgrade_cycle.py",
)


def is_protected_path(relative_path: str) -> bool:
    """Return True when ``relative_path`` is governance-protected."""

    normalized = relative_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    for protected in PROTECTED_PATHS:
        if protected.endswith("/"):
            if normalized.startswith(protected):
                return True
        elif normalized == protected:
            return True
    return False


@dataclass(frozen=True)
class ThanosAuthorization:
    """A persistent, tamper-evident standing authorization record."""

    authorization_id: str
    owner: str
    repository: str
    runtime_scope: str
    enabled: bool
    paused: bool
    created_at: str
    updated_at: str
    allowed_actions: tuple[str, ...]
    authorization_version: int = AUTHORIZATION_VERSION
    authority: str = AUTHORITY
    revoked_at: str | None = None
    revocation_reason: str | None = None
    previous_record_hash: str = GENESIS_HASH
    record_hash: str = ""

    def identity_payload(self) -> dict[str, Any]:
        """Return the hashed portion of the record."""

        return {
            "authorization_id": self.authorization_id,
            "owner": self.owner,
            "repository": self.repository,
            "runtime_scope": self.runtime_scope,
            "authority": self.authority,
            "enabled": self.enabled,
            "paused": self.paused,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason,
            "authorization_version": self.authorization_version,
            "allowed_actions": list(self.allowed_actions),
            "previous_record_hash": self.previous_record_hash,
        }

    def compute_hash(self) -> str:
        """Return the deterministic hash of this record's identity payload."""

        return sha256_payload(self.identity_payload())

    def sealed(self) -> ThanosAuthorization:
        """Return a copy carrying a freshly computed ``record_hash``."""

        return replace(self, record_hash=self.compute_hash())

    def verify(self) -> bool:
        """Return True when ``record_hash`` matches the identity payload."""

        return bool(self.record_hash) and self.record_hash == self.compute_hash()

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def is_active(self) -> bool:
        """True when the grant currently permits autonomous work."""

        return self.enabled and not self.paused and not self.is_revoked

    def to_payload(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["record_hash"] = self.record_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ThanosAuthorization:
        try:
            return cls(
                authorization_id=str(payload["authorization_id"]),
                owner=str(payload["owner"]),
                repository=str(payload["repository"]),
                runtime_scope=str(payload["runtime_scope"]),
                authority=str(payload.get("authority", AUTHORITY)),
                enabled=bool(payload["enabled"]),
                paused=bool(payload["paused"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                revoked_at=(
                    None if payload.get("revoked_at") is None else str(payload["revoked_at"])
                ),
                revocation_reason=(
                    None
                    if payload.get("revocation_reason") is None
                    else str(payload["revocation_reason"])
                ),
                authorization_version=int(payload["authorization_version"]),
                allowed_actions=tuple(str(a) for a in payload["allowed_actions"]),
                previous_record_hash=str(payload["previous_record_hash"]),
                record_hash=str(payload.get("record_hash", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ThanosTamperError("THANOS authorization payload is malformed") from error


def create_authorization(
    *,
    owner: str = OWNER,
    repository: str = REPOSITORY,
    runtime_scope: str = RUNTIME_SCOPE,
    allowed_actions: Sequence[ThanosAction] | None = None,
    now: str | None = None,
    previous_record_hash: str = GENESIS_HASH,
) -> ThanosAuthorization:
    """Create a new enabled standing authorization."""

    actions = tuple(allowed_actions or DEFAULT_ALLOWED_ACTIONS)
    timestamp = now or utc_now_iso()
    parse_utc_iso(timestamp)
    record = ThanosAuthorization(
        authorization_id=new_identifier("thanos"),
        owner=owner,
        repository=repository,
        runtime_scope=runtime_scope,
        enabled=True,
        paused=False,
        created_at=timestamp,
        updated_at=timestamp,
        allowed_actions=tuple(a.value for a in actions),
        previous_record_hash=previous_record_hash,
    )
    return record.sealed()


def _amended(
    authorization: ThanosAuthorization,
    *,
    now: str | None = None,
    **changes: Any,
) -> ThanosAuthorization:
    timestamp = now or utc_now_iso()
    parse_utc_iso(timestamp)
    record = replace(
        authorization,
        updated_at=timestamp,
        previous_record_hash=authorization.record_hash,
        record_hash="",
        **changes,
    )
    return record.sealed()


def pause_authorization(
    authorization: ThanosAuthorization, *, now: str | None = None
) -> ThanosAuthorization:
    """Pause autonomous work without revoking the standing grant."""

    if authorization.is_revoked:
        raise ThanosRevokedError("cannot pause a revoked authorization")
    return _amended(authorization, paused=True, now=now)


def resume_authorization(
    authorization: ThanosAuthorization, *, now: str | None = None
) -> ThanosAuthorization:
    """Resume a paused standing grant."""

    if authorization.is_revoked:
        raise ThanosRevokedError("cannot resume a revoked authorization")
    return _amended(authorization, paused=False, now=now)


def revoke_authorization(
    authorization: ThanosAuthorization,
    *,
    reason: str,
    now: str | None = None,
) -> ThanosAuthorization:
    """Permanently revoke the standing grant."""

    if not reason.strip():
        raise ValueError("revocation reason is required")
    timestamp = now or utc_now_iso()
    return _amended(
        authorization,
        enabled=False,
        paused=False,
        revoked_at=timestamp,
        revocation_reason=reason.strip(),
        now=timestamp,
    )


def permits(
    authorization: ThanosAuthorization,
    action: ThanosAction | str,
    *,
    repository: str = REPOSITORY,
    runtime_scope: str = RUNTIME_SCOPE,
) -> None:
    """Raise unless ``action`` is inside the standing authorization's scope.

    The grant is never consumed: repeated calls with the same action succeed
    for as long as the authorization stays active.
    """

    if not authorization.verify():
        raise ThanosTamperError("THANOS authorization failed hash verification")
    if authorization.is_revoked:
        raise ThanosRevokedError(f"THANOS MODE was revoked: {authorization.revocation_reason}")
    if not authorization.enabled:
        raise ThanosNotEnabledError("THANOS MODE is not enabled")
    if authorization.paused:
        raise ThanosPausedError("THANOS MODE is paused")
    if repository != authorization.repository:
        raise ThanosScopeError(
            f"repository {repository!r} is outside THANOS scope {authorization.repository!r}"
        )
    if runtime_scope != authorization.runtime_scope:
        raise ThanosScopeError(
            f"runtime scope {runtime_scope!r} is outside THANOS scope "
            f"{authorization.runtime_scope!r}"
        )

    value = action.value if isinstance(action, ThanosAction) else str(action)
    if value not in authorization.allowed_actions:
        raise ThanosScopeError(f"action {value!r} is not authorized")


class ThanosAuthorizationStore:
    """Atomic, tamper-evident persistence for the standing authorization."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> ThanosAuthorization | None:
        """Load and verify the persisted chain head, or None when absent."""

        if not self.exists():
            return None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ThanosTamperError("THANOS store is unreadable") from error
        if not isinstance(raw, dict) or "chain" not in raw:
            raise ThanosTamperError("THANOS store is malformed")
        chain = raw["chain"]
        if not isinstance(chain, list) or not chain:
            raise ThanosTamperError("THANOS store contains no records")

        previous = GENESIS_HASH
        record: ThanosAuthorization | None = None
        for entry in chain:
            if not isinstance(entry, dict):
                raise ThanosTamperError("THANOS chain entry is malformed")
            record = ThanosAuthorization.from_payload(entry)
            if not record.verify():
                raise ThanosTamperError(
                    f"THANOS record {record.authorization_id} failed hash verification"
                )
            if record.previous_record_hash != previous:
                raise ThanosTamperError("THANOS chain linkage is broken")
            previous = record.record_hash
        return record

    def history(self) -> tuple[ThanosAuthorization, ...]:
        """Return the verified chain, oldest first."""

        if not self.exists():
            return ()
        self.load()  # verification pass
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return tuple(ThanosAuthorization.from_payload(entry) for entry in raw["chain"])

    def append(self, authorization: ThanosAuthorization) -> ThanosAuthorization:
        """Atomically append a verified record to the chain."""

        if not authorization.verify():
            raise ThanosTamperError("refusing to persist an unsealed record")
        existing = self.history()
        if existing:
            head = existing[-1]
            if authorization.previous_record_hash != head.record_hash:
                raise ThanosTamperError("new record does not link to the current chain head")
        elif authorization.previous_record_hash != GENESIS_HASH:
            raise ThanosTamperError("first record must link to the genesis hash")

        chain = [record.to_payload() for record in existing]
        chain.append(authorization.to_payload())
        self._atomic_write({"version": AUTHORIZATION_VERSION, "chain": chain})
        return authorization

    def _atomic_write(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".thanos-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self._path))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise


def render_status(
    authorization: ThanosAuthorization | None,
    *,
    target_version: str = "2.0.0-beta.1",
) -> str:
    """Render the THANOS status block."""

    if authorization is None:
        return "\n".join(
            (
                "THANOS_MODE=DISABLED",
                f"OWNER={OWNER}",
                f"REPOSITORY={REPOSITORY}",
                "STANDING_AUTHORITY=ABSENT",
                f"TARGET_VERSION={target_version}",
            )
        )

    if authorization.is_revoked:
        state = "REVOKED"
        authority = "REVOKED"
    elif authorization.paused:
        state = "PAUSED"
        authority = "VALID"
    elif authorization.enabled:
        state = "ENABLED"
        authority = "VALID"
    else:
        state = "DISABLED"
        authority = "INACTIVE"

    active = "ENABLED" if authorization.is_active else "SUSPENDED"
    lines = [
        f"THANOS_MODE={state}",
        f"OWNER={authorization.owner}",
        f"REPOSITORY={authorization.repository}",
        f"RUNTIME_SCOPE={authorization.runtime_scope}",
        f"STANDING_AUTHORITY={authority}",
        "PERSISTENT_UNTIL_REVOKED=YES",
        f"INTERNET_RESEARCH={active}",
        f"AUTONOMOUS_SELF_REPAIR={active}",
        f"AUTONOMOUS_COMMIT={active}",
        f"AUTONOMOUS_PUSH={active}",
        f"AUTONOMOUS_PR={active}",
        f"AUTONOMOUS_CI_REPAIR={active}",
        f"AUTONOMOUS_MERGE_WHEN_GREEN={active}",
        f"AUTONOMOUS_RESTART={active}",
        f"AUTONOMOUS_ROLLBACK={active}",
        "PER_STAGE_APPROVAL_PROMPTS=0",
        "OWNER_MERGE_CHECKPOINTS_PER_CYCLE=0",
        f"AUTHORIZATION_ID={authorization.authorization_id}",
        f"RECORD_HASH={authorization.record_hash[:16]}",
        f"TARGET_VERSION={target_version}",
    ]
    if authorization.is_revoked:
        lines.append(f"REVOKED_AT={authorization.revoked_at}")
        lines.append(f"REVOCATION_REASON={authorization.revocation_reason}")
    return "\n".join(lines)


def protected_paths_in(paths: Iterable[str]) -> tuple[str, ...]:
    """Return the governance-protected entries within ``paths``."""

    return tuple(path for path in paths if is_protected_path(path))


# --------------------------------------------------------------------------
# Autonomous merge gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MergePreconditions:
    """Observed facts a cycle must present before an autonomous merge."""

    repository: str
    changed_paths: tuple[str, ...]
    expected_head_sha: str
    actual_head_sha: str
    required_checks_complete: bool
    required_checks_passed: bool
    tested_artifact_sha: str
    proposed_artifact_sha: str
    base_sha_at_test_time: str
    base_sha_now: str
    secrets_detected: bool = False
    rollback_available: bool = True


@dataclass(frozen=True)
class MergeDecision:
    """Result of evaluating the autonomous merge gate."""

    allowed: bool
    blocking_reasons: tuple[str, ...]
    protected_paths: tuple[str, ...]

    @property
    def requires_owner_review(self) -> bool:
        return bool(self.protected_paths)


def evaluate_merge_gate(
    authorization: ThanosAuthorization,
    preconditions: MergePreconditions,
    *,
    runtime_scope: str = RUNTIME_SCOPE,
) -> MergeDecision:
    """Decide whether a cycle may squash-merge its own pull request.

    Autonomous merge is permitted when the standing authorization is active
    and every precondition holds. It is withheld when the change touches a
    governance path, because those files define the validation profile that
    would be certifying the change.
    """

    reasons: list[str] = []

    try:
        permits(
            authorization,
            ThanosAction.MERGE,
            repository=preconditions.repository,
            runtime_scope=runtime_scope,
        )
    except ThanosError as error:
        reasons.append(str(error))

    if preconditions.expected_head_sha != preconditions.actual_head_sha:
        reasons.append("pull-request head moved since the candidate was tested")
    if not preconditions.required_checks_complete:
        reasons.append("required checks have not completed")
    if not preconditions.required_checks_passed:
        reasons.append("required checks did not pass")
    if preconditions.tested_artifact_sha != preconditions.proposed_artifact_sha:
        reasons.append("tested artifact does not match the proposed artifact")
    if preconditions.base_sha_at_test_time != preconditions.base_sha_now:
        reasons.append("base branch advanced; unrelated work would be overwritten")
    if preconditions.secrets_detected:
        reasons.append("secret material detected in the change")
    if not preconditions.rollback_available:
        reasons.append("no last-known-good state is available for rollback")

    protected = protected_paths_in(preconditions.changed_paths)
    if protected:
        reasons.append(
            "change touches governance paths that define its own validation "
            "profile: " + ", ".join(protected)
        )

    return MergeDecision(
        allowed=not reasons,
        blocking_reasons=tuple(reasons),
        protected_paths=protected,
    )
