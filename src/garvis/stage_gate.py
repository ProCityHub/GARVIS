from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Final
from uuid import uuid4

CREATOR: Final = "Adrien D. Thomas"
SYSTEM_DESIGNATION: Final = "GARVIS AGI Beta"
DEVELOPMENT_TRACK: Final = "UPGRADE 2"
SCIENTIFIC_VALIDATION_STATUS: Final = (
    "Full AGI is a development objective; scientific validation is not established."
)

DEFAULT_TRANSITION_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_ACTION_TTL_SECONDS: Final = 10 * 60


class Stage(str, Enum):
    RESEARCH = "research"
    SPECIFICATION = "specification"
    PROTOTYPE = "prototype"
    TESTS = "tests"
    SECURITY_REVIEW = "security_review"
    PR = "pr"
    MERGE = "merge"
    DEPLOYMENT = "deployment"
    PROTOTYPE_REMEDIATION = "prototype_remediation"
    ROLLBACK_REVIEW = "rollback_review"


class ProtectedAction(str, Enum):
    INSTALLATION = "installation"
    DEPENDENCY_CHANGE = "dependency_change"
    PURCHASE = "purchase"
    EXTERNAL_COMMUNICATION = "external_communication"
    FILE_DELETION = "file_deletion"
    LOCAL_BRANCH_DELETION = "local_branch_deletion"
    REMOTE_BRANCH_DELETION = "remote_branch_deletion"
    PROTECTED_SYSTEM_MODIFICATION = "protected_system_modification"
    ACCOUNT_MODIFICATION = "account_modification"
    PUSH = "push"
    FORCE_PUSH = "force_push"
    PULL_REQUEST_PUBLICATION = "pull_request_publication"
    MERGE = "merge"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    SECRET_OPERATION = "secret_operation"
    COMPUTER_USE = "computer_use"
    DESTRUCTIVE_DATABASE_OPERATION = "destructive_database_operation"
    IRREVERSIBLE_MIGRATION = "irreversible_migration"


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


class StageGateError(RuntimeError):
    """Base error for stage-gate policy failures."""


class InvalidTransitionError(StageGateError):
    """Raised when a requested governance transition is not legal."""


class InvalidDecisionError(StageGateError):
    """Raised when a response is not an explicit approval or denial."""


NORMAL_TRANSITIONS: Final[Mapping[Stage, Stage]] = {
    Stage.RESEARCH: Stage.SPECIFICATION,
    Stage.SPECIFICATION: Stage.PROTOTYPE,
    Stage.PROTOTYPE: Stage.TESTS,
    Stage.TESTS: Stage.SECURITY_REVIEW,
    Stage.SECURITY_REVIEW: Stage.PR,
    Stage.PR: Stage.MERGE,
    Stage.MERGE: Stage.DEPLOYMENT,
}

REMEDIATION_TRANSITIONS: Final[Mapping[Stage, frozenset[Stage]]] = {
    Stage.TESTS: frozenset({Stage.PROTOTYPE_REMEDIATION}),
    Stage.SECURITY_REVIEW: frozenset({Stage.PROTOTYPE_REMEDIATION}),
    Stage.PR: frozenset({Stage.PROTOTYPE_REMEDIATION}),
    Stage.PROTOTYPE_REMEDIATION: frozenset({Stage.TESTS}),
    Stage.DEPLOYMENT: frozenset({Stage.ROLLBACK_REVIEW}),
}


def utc_now_iso() -> str:
    """Return the current UTC time in a stable ISO-8601 representation."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_identifier(prefix: str) -> str:
    """Create a non-secret unique identifier with a readable prefix."""

    clean_prefix = prefix.strip().lower().replace(" ", "_")
    if not clean_prefix:
        raise ValueError("identifier prefix must not be empty")
    return f"{clean_prefix}_{uuid4().hex}"


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a record deterministically for hashing and verification."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_payload(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 fingerprint of a canonical governance record."""

    encoded = canonical_json(payload).encode("utf-8")
    return sha256(encoded).hexdigest()


def parse_direct_decision(answer: str) -> Decision:
    """Interpret one direct answer without inferring broad permission."""

    normalized = " ".join(answer.strip().casefold().split())

    if normalized in {"yes", "y", "approve", "approved"}:
        return Decision.APPROVE
    if normalized in {"no", "n", "deny", "denied", "stop"}:
        return Decision.DENY

    raise InvalidDecisionError("an explicit yes, no, approve, or deny decision is required")


def transition_is_legal(source: Stage, destination: Stage) -> bool:
    """Return whether the requested transition is explicitly permitted."""

    if NORMAL_TRANSITIONS.get(source) == destination:
        return True
    return destination in REMEDIATION_TRANSITIONS.get(source, frozenset())


def require_legal_transition(source: Stage, destination: Stage) -> None:
    """Fail closed when a stage is skipped or an unknown path is requested."""

    if not transition_is_legal(source, destination):
        raise InvalidTransitionError(
            f"illegal stage transition: {source.value} -> {destination.value}"
        )


def capability_claim_status(
    *,
    scientifically_validated: bool = False,
) -> dict[str, object]:
    """Return an honest AGI Beta capability-claim status."""

    return {
        "creator": CREATOR,
        "designation": SYSTEM_DESIGNATION,
        "development_track": DEVELOPMENT_TRACK,
        "scientifically_validated_agi": scientifically_validated,
        "scientific_validation_status": (
            "Scientific AGI validation has been established."
            if scientifically_validated
            else SCIENTIFIC_VALIDATION_STATUS
        ),
    }


# GARVIS STAGE-GATE PROTOTYPE PART 1A COMPLETE


@dataclass(frozen=True)
class ProjectRecord:
    """Identity and current governance position for one GARVIS project."""

    project_id: str
    name: str
    repository: str
    worktree: str
    branch: str
    base_commit: str
    artifact_hash: str
    approved_files: tuple[str, ...]
    current_stage: Stage = Stage.RESEARCH
    creator: str = CREATOR
    designation: str = SYSTEM_DESIGNATION
    development_track: str = DEVELOPMENT_TRACK
    scientific_validation_status: str = SCIENTIFIC_VALIDATION_STATUS
    pending_gate: str = "research -> specification"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "repository": self.repository,
            "worktree": self.worktree,
            "branch": self.branch,
            "base_commit": self.base_commit,
            "artifact_hash": self.artifact_hash,
            "approved_files": list(self.approved_files),
            "current_stage": self.current_stage.value,
            "creator": self.creator,
            "designation": self.designation,
            "development_track": self.development_track,
            "scientific_validation_status": self.scientific_validation_status,
            "pending_gate": self.pending_gate,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ApprovalQuestion:
    """One exact question to which Adrien may answer yes or no."""

    question_id: str
    project_id: str
    request_kind: str
    requested_value: str
    explanation: str
    target: str
    scope: tuple[str, ...]
    repository: str
    branch: str
    commit_or_artifact_hash: str
    environment: str = "local"
    created_at: str = field(default_factory=utc_now_iso)
    expires_at: str | None = None

    def identity_payload(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "project_id": self.project_id,
            "request_kind": self.request_kind,
            "requested_value": self.requested_value,
            "explanation": self.explanation,
            "target": self.target,
            "scope": list(self.scope),
            "repository": self.repository,
            "branch": self.branch,
            "commit_or_artifact_hash": self.commit_or_artifact_hash,
            "environment": self.environment,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @property
    def text_hash(self) -> str:
        return sha256_payload(self.identity_payload())


@dataclass(frozen=True)
class AuthorizationGrant:
    """A transition, protected-action, or scoped-exception decision."""

    grant_id: str
    project_id: str
    question_id: str
    question_text_hash: str
    grant_type: str
    requested_value: str
    decision: Decision
    target: str
    scope: tuple[str, ...]
    repository: str
    branch: str
    commit_or_artifact_hash: str
    approver: str = CREATOR
    environment: str = "local"
    one_time: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    expires_at: str | None = None
    consumed_at: str | None = None
    revoked: bool = False
    revocation_reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "grant_id": self.grant_id,
            "project_id": self.project_id,
            "question_id": self.question_id,
            "question_text_hash": self.question_text_hash,
            "grant_type": self.grant_type,
            "requested_value": self.requested_value,
            "decision": self.decision.value,
            "target": self.target,
            "scope": list(self.scope),
            "repository": self.repository,
            "branch": self.branch,
            "commit_or_artifact_hash": self.commit_or_artifact_hash,
            "approver": self.approver,
            "environment": self.environment,
            "one_time": self.one_time,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "consumed_at": self.consumed_at,
            "revoked": self.revoked,
            "revocation_reason": self.revocation_reason,
        }


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence bound to an exact project stage and artifact."""

    evidence_id: str
    project_id: str
    stage: Stage
    evidence_type: str
    artifact_hash: str
    result: str
    command_or_review: str
    environment: str = "local"
    limitations: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now_iso)

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "project_id": self.project_id,
            "stage": self.stage.value,
            "evidence_type": self.evidence_type,
            "artifact_hash": self.artifact_hash,
            "result": self.result,
            "command_or_review": self.command_or_review,
            "environment": self.environment,
            "limitations": list(self.limitations),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AuditEnvelope:
    """Tamper-evident wrapper linking a record to the prior audit record."""

    record_type: str
    payload: Mapping[str, Any]
    previous_record_hash: str
    record_hash: str

    @classmethod
    def seal(
        cls,
        record_type: str,
        payload: Mapping[str, Any],
        previous_record_hash: str = "",
    ) -> AuditEnvelope:
        unsigned = {
            "record_type": record_type,
            "payload": dict(payload),
            "previous_record_hash": previous_record_hash,
        }
        return cls(
            record_type=record_type,
            payload=dict(payload),
            previous_record_hash=previous_record_hash,
            record_hash=sha256_payload(unsigned),
        )

    def verify(self) -> bool:
        unsigned = {
            "record_type": self.record_type,
            "payload": dict(self.payload),
            "previous_record_hash": self.previous_record_hash,
        }
        return self.record_hash == sha256_payload(unsigned)

    def to_payload(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "payload": dict(self.payload),
            "previous_record_hash": self.previous_record_hash,
            "record_hash": self.record_hash,
        }


# GARVIS STAGE-GATE PROTOTYPE PART 1B COMPLETE


class AuthorizationError(StageGateError):
    """Base error for invalid or unusable authorization."""


class AuthorizationMismatchError(AuthorizationError):
    """Raised when approval scope does not match the requested operation."""


class AuthorizationExpiredError(AuthorizationError):
    """Raised when an approval or question has expired."""


class AuthorizationConsumedError(AuthorizationError):
    """Raised when a one-time approval has already been used."""


class AuthorizationRevokedError(AuthorizationError):
    """Raised when Adrien has revoked an approval."""


class AuthorizationDeniedError(AuthorizationError):
    """Raised when the recorded decision is a denial."""


def parse_utc_iso(value: str) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp and normalize it to UTC."""

    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")

    return parsed.astimezone(timezone.utc)


def expiry_iso(
    ttl_seconds: int,
    *,
    now: datetime | None = None,
) -> str:
    """Return a UTC expiration timestamp for a positive time-to-live."""

    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be positive")

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must include a timezone")

    expires = current.astimezone(timezone.utc) + timedelta(seconds=ttl_seconds)
    return expires.isoformat().replace("+00:00", "Z")


def pending_gate_for(stage: Stage) -> str:
    """Return the next normal gate in plain machine-readable form."""

    destination = NORMAL_TRANSITIONS.get(stage)
    if destination is not None:
        return f"{stage.value} -> {destination.value}"
    if stage is Stage.PROTOTYPE_REMEDIATION:
        return "prototype_remediation -> tests"
    if stage is Stage.ROLLBACK_REVIEW:
        return "rollback decision pending"
    return "no automatic transition"


def create_transition_question(
    project: ProjectRecord,
    destination: Stage,
    explanation: str,
    *,
    ttl_seconds: int = DEFAULT_TRANSITION_TTL_SECONDS,
) -> ApprovalQuestion:
    """Create one exact stage-transition question for Adrien."""

    require_legal_transition(project.current_stage, destination)

    clean_explanation = explanation.strip()
    if not clean_explanation:
        raise ValueError("transition explanation must not be empty")

    return ApprovalQuestion(
        question_id=new_identifier("question"),
        project_id=project.project_id,
        request_kind="transition",
        requested_value=f"{project.current_stage.value}->{destination.value}",
        explanation=clean_explanation,
        target=project.name,
        scope=project.approved_files,
        repository=project.repository,
        branch=project.branch,
        commit_or_artifact_hash=project.artifact_hash,
        expires_at=expiry_iso(ttl_seconds),
    )


def create_protected_action_question(
    project: ProjectRecord,
    action: ProtectedAction,
    target: str,
    explanation: str,
    *,
    scope: tuple[str, ...] | None = None,
    environment: str = "local",
    ttl_seconds: int = DEFAULT_ACTION_TTL_SECONDS,
) -> ApprovalQuestion:
    """Create one exact protected-action question for Adrien."""

    clean_target = target.strip()
    clean_explanation = explanation.strip()
    clean_environment = environment.strip()

    if not clean_target:
        raise ValueError("protected-action target must not be empty")
    if not clean_explanation:
        raise ValueError("protected-action explanation must not be empty")
    if not clean_environment:
        raise ValueError("environment must not be empty")

    return ApprovalQuestion(
        question_id=new_identifier("question"),
        project_id=project.project_id,
        request_kind="protected_action",
        requested_value=action.value,
        explanation=clean_explanation,
        target=clean_target,
        scope=scope if scope is not None else project.approved_files,
        repository=project.repository,
        branch=project.branch,
        commit_or_artifact_hash=project.artifact_hash,
        environment=clean_environment,
        expires_at=expiry_iso(ttl_seconds),
    )


def grant_from_answer(
    question: ApprovalQuestion,
    answer: str,
    *,
    one_time: bool = True,
    approver: str = CREATOR,
) -> AuthorizationGrant:
    """Bind one direct answer to the exact stored approval question."""

    clean_approver = approver.strip()
    if clean_approver != CREATOR:
        raise AuthorizationMismatchError(f"approval authority must be {CREATOR}")

    return AuthorizationGrant(
        grant_id=new_identifier("grant"),
        project_id=question.project_id,
        question_id=question.question_id,
        question_text_hash=question.text_hash,
        grant_type=question.request_kind,
        requested_value=question.requested_value,
        decision=parse_direct_decision(answer),
        target=question.target,
        scope=question.scope,
        repository=question.repository,
        branch=question.branch,
        commit_or_artifact_hash=question.commit_or_artifact_hash,
        approver=clean_approver,
        environment=question.environment,
        one_time=one_time,
        expires_at=question.expires_at,
    )


def validate_grant(
    question: ApprovalQuestion,
    grant: AuthorizationGrant,
    *,
    now: datetime | None = None,
) -> None:
    """Fail closed unless a grant matches the exact approval question."""

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    expected = {
        "project_id": question.project_id,
        "question_id": question.question_id,
        "question_text_hash": question.text_hash,
        "grant_type": question.request_kind,
        "requested_value": question.requested_value,
        "target": question.target,
        "scope": question.scope,
        "repository": question.repository,
        "branch": question.branch,
        "commit_or_artifact_hash": question.commit_or_artifact_hash,
        "environment": question.environment,
        "approver": CREATOR,
    }

    mismatches = [
        name for name, expected_value in expected.items() if getattr(grant, name) != expected_value
    ]
    if mismatches:
        raise AuthorizationMismatchError("authorization does not match: " + ", ".join(mismatches))

    if grant.decision is not Decision.APPROVE:
        raise AuthorizationDeniedError("Adrien denied this request")

    if grant.revoked:
        raise AuthorizationRevokedError(grant.revocation_reason or "authorization was revoked")

    if grant.one_time and grant.consumed_at is not None:
        raise AuthorizationConsumedError("one-time authorization has already been consumed")

    for label, value in (
        ("question", question.expires_at),
        ("authorization", grant.expires_at),
    ):
        if value is not None and parse_utc_iso(value) <= current:
            raise AuthorizationExpiredError(f"{label} has expired")


def consume_grant(
    question: ApprovalQuestion,
    grant: AuthorizationGrant,
    *,
    now: datetime | None = None,
) -> AuthorizationGrant:
    """Validate and atomically represent consumption of a one-time grant."""

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    validate_grant(question, grant, now=current)

    if not grant.one_time:
        return grant

    return replace(
        grant,
        consumed_at=current.isoformat().replace("+00:00", "Z"),
    )


def apply_transition(
    project: ProjectRecord,
    destination: Stage,
    question: ApprovalQuestion,
    grant: AuthorizationGrant,
    *,
    now: datetime | None = None,
) -> tuple[ProjectRecord, AuthorizationGrant]:
    """Apply one approved legal transition and consume its authorization."""

    require_legal_transition(project.current_stage, destination)

    expected_request = f"{project.current_stage.value}->{destination.value}"
    if question.request_kind != "transition":
        raise AuthorizationMismatchError("approval question is not a transition request")
    if question.requested_value != expected_request:
        raise AuthorizationMismatchError("approval question targets a different transition")
    if question.project_id != project.project_id:
        raise AuthorizationMismatchError("approval question targets a different project")
    if question.repository != project.repository:
        raise AuthorizationMismatchError("approval question targets a different repository")
    if question.branch != project.branch:
        raise AuthorizationMismatchError("approval question targets a different branch")
    if question.commit_or_artifact_hash != project.artifact_hash:
        raise AuthorizationMismatchError("approval question targets a different artifact")
    if question.scope != project.approved_files:
        raise AuthorizationMismatchError("approval question targets a different file scope")

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    consumed = consume_grant(question, grant, now=current)

    updated = replace(
        project,
        current_stage=destination,
        pending_gate=pending_gate_for(destination),
        updated_at=current.isoformat().replace("+00:00", "Z"),
    )
    return updated, consumed


def consume_protected_action_authorization(
    question: ApprovalQuestion,
    grant: AuthorizationGrant,
    *,
    now: datetime | None = None,
) -> AuthorizationGrant:
    """Validate a protected-action grant without executing the action."""

    if question.request_kind != "protected_action":
        raise AuthorizationMismatchError("approval question is not a protected-action request")
    return consume_grant(question, grant, now=now)


# GARVIS STAGE-GATE PROTOTYPE PART 1C COMPLETE


class AuditChainError(StageGateError):
    """Raised when governance history cannot be verified."""


class EvidenceMismatchError(StageGateError):
    """Raised when evidence belongs to another project, stage, or artifact."""


def revoke_grant(
    grant: AuthorizationGrant,
    reason: str,
) -> AuthorizationGrant:
    """Return a revoked authorization record without deleting history."""

    clean_reason = reason.strip()
    if not clean_reason:
        raise ValueError("revocation reason must not be empty")

    return replace(
        grant,
        revoked=True,
        revocation_reason=clean_reason,
    )


def create_evidence(
    project: ProjectRecord,
    evidence_type: str,
    result: str,
    command_or_review: str,
    *,
    stage: Stage | None = None,
    artifact_hash: str | None = None,
    environment: str = "local",
    limitations: tuple[str, ...] = (),
) -> EvidenceRecord:
    """Create evidence explicitly bound to a project stage and artifact."""

    clean_type = evidence_type.strip()
    clean_result = result.strip()
    clean_command = command_or_review.strip()
    clean_environment = environment.strip()

    if not clean_type:
        raise ValueError("evidence_type must not be empty")
    if not clean_result:
        raise ValueError("result must not be empty")
    if not clean_command:
        raise ValueError("command_or_review must not be empty")
    if not clean_environment:
        raise ValueError("environment must not be empty")

    return EvidenceRecord(
        evidence_id=new_identifier("evidence"),
        project_id=project.project_id,
        stage=stage or project.current_stage,
        evidence_type=clean_type,
        artifact_hash=artifact_hash or project.artifact_hash,
        result=clean_result,
        command_or_review=clean_command,
        environment=clean_environment,
        limitations=tuple(item.strip() for item in limitations if item.strip()),
    )


def validate_evidence(
    project: ProjectRecord,
    evidence: EvidenceRecord,
    *,
    expected_stage: Stage | None = None,
) -> None:
    """Fail closed when evidence does not match the exact governed artifact."""

    stage = expected_stage or project.current_stage
    mismatches: list[str] = []

    if evidence.project_id != project.project_id:
        mismatches.append("project_id")
    if evidence.stage is not stage:
        mismatches.append("stage")
    if evidence.artifact_hash != project.artifact_hash:
        mismatches.append("artifact_hash")

    if mismatches:
        raise EvidenceMismatchError("evidence does not match: " + ", ".join(mismatches))


def seal_audit_records(
    records: tuple[tuple[str, Mapping[str, Any]], ...],
) -> tuple[AuditEnvelope, ...]:
    """Seal ordered records into one deterministic hash-linked chain."""

    previous_hash = ""
    envelopes: list[AuditEnvelope] = []

    for record_type, payload in records:
        clean_type = record_type.strip()
        if not clean_type:
            raise ValueError("audit record type must not be empty")

        envelope = AuditEnvelope.seal(
            clean_type,
            payload,
            previous_hash,
        )
        envelopes.append(envelope)
        previous_hash = envelope.record_hash

    return tuple(envelopes)


def verify_audit_chain(
    envelopes: tuple[AuditEnvelope, ...],
) -> bool:
    """Verify record hashes, ordering, linkage, and duplicate identities."""

    expected_previous_hash = ""
    seen_record_hashes: set[str] = set()
    seen_identifiers: set[tuple[str, str]] = set()

    for index, envelope in enumerate(envelopes):
        if envelope.previous_record_hash != expected_previous_hash:
            raise AuditChainError(f"audit linkage failed at record {index}")

        if not envelope.verify():
            raise AuditChainError(f"audit record hash failed at record {index}")

        if envelope.record_hash in seen_record_hashes:
            raise AuditChainError(f"duplicate audit record detected at record {index}")
        seen_record_hashes.add(envelope.record_hash)

        payload = dict(envelope.payload)
        primary_identifier_fields = {
            "project": "project_id",
            "approval_question": "question_id",
            "authorization_grant": "grant_id",
            "evidence": "evidence_id",
        }
        identifier_field = primary_identifier_fields.get(envelope.record_type)

        if identifier_field is not None:
            value = payload.get(identifier_field)
            if not isinstance(value, str) or not value:
                raise AuditChainError(f"missing {identifier_field} at record {index}")

            identity = (identifier_field, value)
            if identity in seen_identifiers:
                raise AuditChainError(f"duplicate {identifier_field} detected at record {index}")
            seen_identifiers.add(identity)

        expected_previous_hash = envelope.record_hash

    return True


def active_authorization_count(
    grants: tuple[AuthorizationGrant, ...],
) -> int:
    """Count approved grants that are not revoked or already consumed."""

    return sum(
        1
        for grant in grants
        if grant.decision is Decision.APPROVE and not grant.revoked and grant.consumed_at is None
    )


def render_project_status(
    project: ProjectRecord,
    *,
    grants: tuple[AuthorizationGrant, ...] = (),
    evidence: tuple[EvidenceRecord, ...] = (),
    detailed: bool = False,
) -> str:
    """Render understandable status without implying protected permission."""

    active = active_authorization_count(grants)
    revoked = sum(1 for grant in grants if grant.revoked)
    consumed = sum(1 for grant in grants if grant.consumed_at is not None)

    lines = [
        f"System: {project.designation}",
        f"Creator and approval authority: {project.creator}",
        f"Development track: {project.development_track}",
        f"Current stage: {project.current_stage.value}",
        f"Pending gate: {project.pending_gate}",
        f"Scientific status: {project.scientific_validation_status}",
        f"Active authorizations: {active}",
        "Protected actions remain separately controlled.",
    ]

    if not detailed:
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Detailed governance identity:",
            f"- Project ID: {project.project_id}",
            f"- Project: {project.name}",
            f"- Repository: {project.repository}",
            f"- Worktree: {project.worktree}",
            f"- Branch: {project.branch}",
            f"- Base commit: {project.base_commit}",
            f"- Artifact hash: {project.artifact_hash}",
            f"- Revoked authorizations: {revoked}",
            f"- Consumed authorizations: {consumed}",
            f"- Evidence records: {len(evidence)}",
            "",
            "Approved file boundary:",
        ]
    )

    lines.extend(f"- {item}" for item in project.approved_files)

    lines.extend(
        [
            "",
            "Still requires separate Adrien approval:",
            "- Installation or dependency changes",
            "- Push or force-push",
            "- Pull-request publication",
            "- Merge",
            "- Local or remote branch deletion",
            "- External communication or purchases",
            "- Protected-system modification",
            "- Deployment or rollback",
        ]
    )

    return "\n".join(lines)


# GARVIS STAGE-GATE PROTOTYPE PART 1D COMPLETE
