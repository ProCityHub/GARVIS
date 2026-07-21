"""Epistemic and operational filing for GARVIS.

Uncertain claims are preserved for review instead of being mislabeled as
software errors. Operational errors and epistemic claims remain separate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClaimDomain(str, Enum):
    REPOSITORY = "repository"
    SCIENTIFIC = "scientific"
    IDENTITY = "identity"
    SYMBOLIC = "symbolic"
    OPERATIONAL = "operational"
    GENERAL = "general"


class EpistemicStatus(str, Enum):
    VERIFIED = "verified"
    SUPPORTED = "supported"
    PROVISIONAL = "provisional"
    HYPOTHESIS = "hypothesis"
    SPECULATIVE = "speculative"
    SYMBOLIC = "symbolic"
    ANOMALY = "anomaly"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"
    IDENTITY_DRAFT = "identity_draft"


class ReviewState(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ErrorCategory(str, Enum):
    SYNTAX = "syntax"
    TEST = "test"
    TYPECHECK = "typecheck"
    LINT = "lint"
    RUNTIME = "runtime"
    DATA = "data"
    SECURITY = "security"
    GOVERNANCE = "governance"
    UNKNOWN = "unknown"


class ErrorStatus(str, Enum):
    OPEN = "open"
    TRIAGED = "triaged"
    FIX_IN_PROGRESS = "fix_in_progress"
    RESOLVED = "resolved"
    DUPLICATE = "duplicate"
    WONT_FIX = "wont_fix"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    summary: str
    reproducible: bool = False
    weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("evidence source must not be empty")
        if not self.summary.strip():
            raise ValueError("evidence summary must not be empty")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("evidence weight must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RevisionEntry:
    timestamp: str
    actor: str
    reason: str
    previous_status: str | None = None
    new_status: str | None = None


@dataclass
class ClaimRecord:
    statement: str
    domain: ClaimDomain
    status: EpistemicStatus
    scope: str
    confidence: float
    created_by: str
    permitted_wording: str
    prohibited_wording: str = ""
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)
    review_state: ReviewState = ReviewState.OPEN
    claim_id: str = field(default_factory=lambda: f"CLM-{uuid4().hex[:12].upper()}")
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    revision_history: list[RevisionEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("claim statement must not be empty")
        if not self.scope.strip():
            raise ValueError("claim scope must not be empty")
        if not self.created_by.strip():
            raise ValueError("created_by must not be empty")
        if not self.permitted_wording.strip():
            raise ValueError("permitted_wording must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.status is EpistemicStatus.VERIFIED and not self.supporting_evidence:
            raise ValueError("verified claims require supporting evidence")

    @property
    def can_be_stated_as_fact(self) -> bool:
        return (
            self.status is EpistemicStatus.VERIFIED
            and bool(self.supporting_evidence)
            and not self.contradicting_evidence
            and self.review_state not in {ReviewState.REJECTED, ReviewState.ARCHIVED}
        )

    @property
    def can_be_stated_as_scientific_fact(self) -> bool:
        return (
            self.domain is ClaimDomain.SCIENTIFIC
            and self.can_be_stated_as_fact
            and any(item.reproducible for item in self.supporting_evidence)
        )

    def rendered_statement(self) -> str:
        if self.can_be_stated_as_fact:
            return self.statement
        return f"[{self.status.value.upper()}] {self.permitted_wording}"

    def add_evidence(self, item: EvidenceItem, *, contradicts: bool = False) -> None:
        target = self.contradicting_evidence if contradicts else self.supporting_evidence
        target.append(item)
        self.updated_at = _utc_now()

    def transition(
        self,
        new_status: EpistemicStatus,
        *,
        actor: str,
        reason: str,
        confidence: float | None = None,
    ) -> None:
        if not actor.strip():
            raise ValueError("transition actor must not be empty")
        if not reason.strip():
            raise ValueError("transition reason must not be empty")
        if new_status is EpistemicStatus.VERIFIED and not self.supporting_evidence:
            raise ValueError("cannot verify a claim without supporting evidence")
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            self.confidence = confidence
        old_status = self.status
        self.status = new_status
        self.updated_at = _utc_now()
        self.revision_history.append(
            RevisionEntry(
                timestamp=self.updated_at,
                actor=actor,
                reason=reason,
                previous_status=old_status.value,
                new_status=new_status.value,
            )
        )


@dataclass
class OperationalErrorRecord:
    message: str
    category: ErrorCategory
    source: str
    created_by: str
    status: ErrorStatus = ErrorStatus.OPEN
    details: str = ""
    related_claim_ids: list[str] = field(default_factory=list)
    error_id: str = field(default_factory=lambda: f"ERR-{uuid4().hex[:12].upper()}")
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    revision_history: list[RevisionEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("error message must not be empty")
        if not self.source.strip():
            raise ValueError("error source must not be empty")
        if not self.created_by.strip():
            raise ValueError("created_by must not be empty")

    def transition(self, new_status: ErrorStatus, *, actor: str, reason: str) -> None:
        if not actor.strip():
            raise ValueError("transition actor must not be empty")
        if not reason.strip():
            raise ValueError("transition reason must not be empty")
        old_status = self.status
        self.status = new_status
        self.updated_at = _utc_now()
        self.revision_history.append(
            RevisionEntry(
                timestamp=self.updated_at,
                actor=actor,
                reason=reason,
                previous_status=old_status.value,
                new_status=new_status.value,
            )
        )


@dataclass
class FilingSystem:
    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    errors: dict[str, OperationalErrorRecord] = field(default_factory=dict)

    def file_claim(self, record: ClaimRecord) -> ClaimRecord:
        if record.claim_id in self.claims:
            raise ValueError(f"duplicate claim id: {record.claim_id}")
        self.claims[record.claim_id] = record
        return record

    def file_error(self, record: OperationalErrorRecord) -> OperationalErrorRecord:
        if record.error_id in self.errors:
            raise ValueError(f"duplicate error id: {record.error_id}")
        self.errors[record.error_id] = record
        return record

    def claims_by_status(self, status: EpistemicStatus) -> tuple[ClaimRecord, ...]:
        return tuple(record for record in self.claims.values() if record.status is status)

    def errors_by_status(self, status: ErrorStatus) -> tuple[OperationalErrorRecord, ...]:
        return tuple(record for record in self.errors.values() if record.status is status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "claims": [asdict(record) for record in self.claims.values()],
            "errors": [asdict(record) for record in self.errors.values()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FilingSystem":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        system = cls()
        for item in raw.get("claims", []):
            item["domain"] = ClaimDomain(item["domain"])
            item["status"] = EpistemicStatus(item["status"])
            item["review_state"] = ReviewState(item["review_state"])
            item["supporting_evidence"] = [
                EvidenceItem(**evidence) for evidence in item.get("supporting_evidence", [])
            ]
            item["contradicting_evidence"] = [
                EvidenceItem(**evidence) for evidence in item.get("contradicting_evidence", [])
            ]
            item["revision_history"] = [
                RevisionEntry(**entry) for entry in item.get("revision_history", [])
            ]
            system.file_claim(ClaimRecord(**item))
        for item in raw.get("errors", []):
            item["category"] = ErrorCategory(item["category"])
            item["status"] = ErrorStatus(item["status"])
            item["revision_history"] = [
                RevisionEntry(**entry) for entry in item.get("revision_history", [])
            ]
            system.file_error(OperationalErrorRecord(**item))
        return system
