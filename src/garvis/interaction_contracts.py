"""Non-executing GARVIS interaction data contracts.

These contracts represent observations, evidence, proposals, scoped grants,
and execution results. They provide no device, provider, network, filesystem,
or deployment execution capability.

A CapabilityGrant is data describing scoped authorization evidence.
Final protected-action authority remains with GARVIS governance/stage-gate
enforcement; ``grant_authorizes`` is only a pure scope-validation helper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA_VERSION = "garvis.interaction.v1"

_SECRET_KEY_MARKERS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "credential",
)


def _required(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _secret_key(name: object) -> bool:
    lowered = str(name).casefold()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def _freeze(value: Any) -> Any:
    """Recursively isolate caller-owned mutable state."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze(item)
            for key, item in value.items()
        })

    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)

    return value


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _safe(getattr(value, item.name))
            for item in fields(value)
        }

    if isinstance(value, datetime):
        return _utc_iso(value)

    if isinstance(value, Mapping):
        return {
            str(key): _safe(item)
            for key, item in value.items()
            if not _secret_key(key)
        }

    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return sorted((_safe(item) for item in value), key=repr)

    return value


def canonical_json(value: Any) -> str:
    """Return deterministic, secret-filtered JSON suitable for hashing."""
    return json.dumps(
        _safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    source_type: str
    timestamp: datetime
    sequence: int
    acquisition_capability: str
    raw_reference: str
    provenance: str
    trust_classification: str
    content_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "source",
            "source_type",
            "acquisition_capability",
            "raw_reference",
            "provenance",
            "trust_classification",
            "content_type",
        ):
            _required(name, getattr(self, name))

        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")

        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be datetime")

        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_type: str
    source: str
    created_at: datetime
    content: Any
    digest: str
    provenance: str
    verification_status: str
    parent_evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "evidence_type",
            "source",
            "digest",
            "provenance",
            "verification_status",
        ):
            _required(name, getattr(self, name))

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be datetime")

        for item in self.parent_evidence_ids:
            _required("parent_evidence_id", item)

        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class ActionProposal:
    proposal_id: str
    run_id: str
    session_id: str
    operation: str
    target: str
    required_capability: str
    rationale: str
    supporting_evidence_ids: tuple[str, ...]
    expected_consequence: str
    risk_classification: str
    approval_requirement: str
    created_at: datetime
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "run_id",
            "session_id",
            "operation",
            "target",
            "required_capability",
            "rationale",
            "expected_consequence",
            "risk_classification",
            "approval_requirement",
        ):
            _required(name, getattr(self, name))

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be datetime")

        for item in self.supporting_evidence_ids:
            _required("supporting_evidence_id", item)


@dataclass(frozen=True)
class CapabilityGrant:
    grant_id: str
    capability_id: str
    actor: str
    project: str
    stage: str
    operation: str
    target: str
    approval_evidence_id: str
    issued_at: datetime
    expires_at: datetime | None
    revoked: bool
    scope_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "grant_id",
            "capability_id",
            "actor",
            "project",
            "stage",
            "operation",
            "target",
            "approval_evidence_id",
        ):
            _required(name, getattr(self, name))

        if not isinstance(self.issued_at, datetime):
            raise TypeError("issued_at must be datetime")

        if self.expires_at is not None and not isinstance(self.expires_at, datetime):
            raise TypeError("expires_at must be datetime or None")

        if not isinstance(self.revoked, bool):
            raise TypeError("revoked must be bool")

        object.__setattr__(
            self,
            "scope_metadata",
            _freeze(self.scope_metadata),
        )


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    proposal_id: str
    grant_id: str
    adapter_id: str
    started_at: datetime
    completed_at: datetime
    status: str
    result_evidence_ids: tuple[str, ...]
    error: str | None
    observed_state_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "execution_id",
            "proposal_id",
            "grant_id",
            "adapter_id",
            "status",
            "observed_state_digest",
        ):
            _required(name, getattr(self, name))

        if not isinstance(self.started_at, datetime):
            raise TypeError("started_at must be datetime")

        if not isinstance(self.completed_at, datetime):
            raise TypeError("completed_at must be datetime")

        for item in self.result_evidence_ids:
            _required("result_evidence_id", item)

        object.__setattr__(self, "metadata", _freeze(self.metadata))


def grant_authorizes(
    grant: CapabilityGrant,
    proposal: ActionProposal,
    *,
    now: datetime | None = None,
) -> bool:
    """Validate grant scope only; this function never executes an action."""
    if not isinstance(grant, CapabilityGrant):
        return False

    if not isinstance(proposal, ActionProposal):
        return False

    if grant.revoked:
        return False

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    issued = grant.issued_at
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)

    if current < issued:
        return False

    if grant.expires_at is not None:
        expires = grant.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)

        if current >= expires:
            return False

    if grant.capability_id != proposal.required_capability:
        return False

    if grant.operation != proposal.operation:
        return False

    if grant.target != proposal.target:
        return False

    return True
