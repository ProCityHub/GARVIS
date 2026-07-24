from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from garvis.stage_gate import (
    CREATOR,
    ApprovalQuestion,
    AuditChainError,
    AuditEnvelope,
    AuthorizationGrant,
    Decision,
    EvidenceRecord,
    ProjectRecord,
    Stage,
    apply_transition,
    utc_now_iso,
    validate_evidence,
    verify_audit_chain,
)


class StageGateStoreError(RuntimeError):
    """Base error for local stage-gate storage failures."""


class StoreCorruptionError(StageGateStoreError):
    """Raised when persisted governance history cannot be verified."""


class StoreReferenceError(StageGateStoreError):
    """Raised when a record refers to the wrong governed object."""


class StoreConflictError(StageGateStoreError):
    """Raised when the audit store changed before an operation was saved."""


class StageGateStore:
    """Local transaction-safe SQLite store for governance audit records."""

    SCHEMA_VERSION = "1"

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def __enter__(self) -> "StageGateStore":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize_schema(self) -> None:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_record_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

            defaults = {
                "schema_version": self.SCHEMA_VERSION,
                "chain_head": "",
                "chain_length": "0",
            }
            for key, value in defaults.items():
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO metadata(key, value)
                    VALUES (?, ?)
                    """,
                    (key, value),
                )

            schema_version = self._metadata_value("schema_version")
            if schema_version != self.SCHEMA_VERSION:
                raise StoreCorruptionError(
                    "unsupported stage-gate database schema version"
                )

            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _metadata_value(self, key: str) -> str:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            raise StoreCorruptionError(
                f"required database metadata is missing: {key}"
            )
        return str(row["value"])

    def _set_metadata(self, key: str, value: str) -> None:
        cursor = self.connection.execute(
            "UPDATE metadata SET value = ? WHERE key = ?",
            (value, key),
        )
        if cursor.rowcount != 1:
            raise StoreCorruptionError(
                f"required database metadata could not be updated: {key}"
            )

    @property
    def record_count(self) -> int:
        return int(self._metadata_value("chain_length"))

    @property
    def chain_head(self) -> str:
        return self._metadata_value("chain_head")

    def append(
        self,
        record_type: str,
        payload: Mapping[str, Any],
    ) -> AuditEnvelope:
        """Append one audit record and update the chain anchor atomically."""

        clean_type = record_type.strip()
        if not clean_type:
            raise ValueError("record_type must not be empty")

        payload_copy = dict(payload)

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.verify_integrity()

            previous_hash = self._metadata_value("chain_head")
            previous_length = int(self._metadata_value("chain_length"))

            envelope = AuditEnvelope.seal(
                clean_type,
                payload_copy,
                previous_hash,
            )

            self.connection.execute(
                """
                INSERT INTO audit_records(
                    record_type,
                    payload_json,
                    previous_record_hash,
                    record_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    envelope.record_type,
                    json.dumps(
                        dict(envelope.payload),
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    envelope.previous_record_hash,
                    envelope.record_hash,
                    utc_now_iso(),
                ),
            )

            self._set_metadata("chain_head", envelope.record_hash)
            self._set_metadata(
                "chain_length",
                str(previous_length + 1),
            )
            self.connection.commit()
            return envelope
        except Exception:
            self.connection.rollback()
            raise

    def load_chain(self) -> tuple[AuditEnvelope, ...]:
        """Read the complete audit chain in its stored sequence."""

        rows = self.connection.execute(
            """
            SELECT
                record_type,
                payload_json,
                previous_record_hash,
                record_hash
            FROM audit_records
            ORDER BY sequence ASC
            """
        ).fetchall()

        envelopes: list[AuditEnvelope] = []
        for index, row in enumerate(rows):
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError as exc:
                raise StoreCorruptionError(
                    f"invalid audit JSON at record {index}"
                ) from exc

            if not isinstance(payload, dict):
                raise StoreCorruptionError(
                    f"audit payload is not an object at record {index}"
                )

            envelopes.append(
                AuditEnvelope(
                    record_type=str(row["record_type"]),
                    payload=payload,
                    previous_record_hash=str(
                        row["previous_record_hash"]
                    ),
                    record_hash=str(row["record_hash"]),
                )
            )

        return tuple(envelopes)

    def verify_integrity(self) -> bool:
        """Verify chain hashes, record count, ordering, and final anchor."""

        chain = self.load_chain()

        try:
            expected_length = int(
                self._metadata_value("chain_length")
            )
        except ValueError as exc:
            raise StoreCorruptionError(
                "stored chain length is not an integer"
            ) from exc

        if len(chain) != expected_length:
            raise StoreCorruptionError(
                "stored audit-record count does not match its anchor"
            )

        expected_head = self._metadata_value("chain_head")
        actual_head = chain[-1].record_hash if chain else ""

        if actual_head != expected_head:
            raise StoreCorruptionError(
                "stored audit-chain head does not match its anchor"
            )

        try:
            verify_audit_chain(chain)
        except AuditChainError as exc:
            raise StoreCorruptionError(
                f"stored audit chain failed verification: {exc}"
            ) from exc

        return True

    def status(self) -> dict[str, object]:
        """Return local read-only information about the audit store."""

        self.verify_integrity()
        return {
            "database_path": str(self.database_path),
            "schema_version": self._metadata_value("schema_version"),
            "record_count": self.record_count,
            "chain_head": self.chain_head,
            "integrity_verified": True,
        }


    @staticmethod
    def _project_from_payload(
        payload: Mapping[str, Any],
    ) -> ProjectRecord:
        try:
            return ProjectRecord(
                project_id=str(payload["project_id"]),
                name=str(payload["name"]),
                repository=str(payload["repository"]),
                worktree=str(payload["worktree"]),
                branch=str(payload["branch"]),
                base_commit=str(payload["base_commit"]),
                artifact_hash=str(payload["artifact_hash"]),
                approved_files=tuple(
                    str(item) for item in payload["approved_files"]
                ),
                current_stage=Stage(str(payload["current_stage"])),
                creator=str(payload["creator"]),
                designation=str(payload["designation"]),
                development_track=str(payload["development_track"]),
                scientific_validation_status=str(
                    payload["scientific_validation_status"]
                ),
                pending_gate=str(payload["pending_gate"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCorruptionError(
                "stored project record has an invalid structure"
            ) from exc

    @staticmethod
    def _question_from_payload(
        payload: Mapping[str, Any],
    ) -> ApprovalQuestion:
        try:
            expires_value = payload.get("expires_at")
            return ApprovalQuestion(
                question_id=str(payload["question_id"]),
                project_id=str(payload["project_id"]),
                request_kind=str(payload["request_kind"]),
                requested_value=str(payload["requested_value"]),
                explanation=str(payload["explanation"]),
                target=str(payload["target"]),
                scope=tuple(str(item) for item in payload["scope"]),
                repository=str(payload["repository"]),
                branch=str(payload["branch"]),
                commit_or_artifact_hash=str(
                    payload["commit_or_artifact_hash"]
                ),
                environment=str(payload["environment"]),
                created_at=str(payload["created_at"]),
                expires_at=(
                    None
                    if expires_value is None
                    else str(expires_value)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCorruptionError(
                "stored approval question has an invalid structure"
            ) from exc

    @staticmethod
    def _grant_from_payload(
        payload: Mapping[str, Any],
    ) -> AuthorizationGrant:
        try:
            expires_value = payload.get("expires_at")
            consumed_value = payload.get("consumed_at")
            reason_value = payload.get("revocation_reason")

            return AuthorizationGrant(
                grant_id=str(payload["grant_id"]),
                project_id=str(payload["project_id"]),
                question_id=str(payload["question_id"]),
                question_text_hash=str(
                    payload["question_text_hash"]
                ),
                grant_type=str(payload["grant_type"]),
                requested_value=str(payload["requested_value"]),
                decision=Decision(str(payload["decision"])),
                target=str(payload["target"]),
                scope=tuple(str(item) for item in payload["scope"]),
                repository=str(payload["repository"]),
                branch=str(payload["branch"]),
                commit_or_artifact_hash=str(
                    payload["commit_or_artifact_hash"]
                ),
                approver=str(payload["approver"]),
                environment=str(payload["environment"]),
                one_time=bool(payload["one_time"]),
                created_at=str(payload["created_at"]),
                expires_at=(
                    None
                    if expires_value is None
                    else str(expires_value)
                ),
                consumed_at=(
                    None
                    if consumed_value is None
                    else str(consumed_value)
                ),
                revoked=bool(payload["revoked"]),
                revocation_reason=(
                    None
                    if reason_value is None
                    else str(reason_value)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCorruptionError(
                "stored authorization has an invalid structure"
            ) from exc

    @staticmethod
    def _evidence_from_payload(
        payload: Mapping[str, Any],
    ) -> EvidenceRecord:
        try:
            return EvidenceRecord(
                evidence_id=str(payload["evidence_id"]),
                project_id=str(payload["project_id"]),
                stage=Stage(str(payload["stage"])),
                evidence_type=str(payload["evidence_type"]),
                artifact_hash=str(payload["artifact_hash"]),
                result=str(payload["result"]),
                command_or_review=str(payload["command_or_review"]),
                environment=str(payload["environment"]),
                limitations=tuple(
                    str(item) for item in payload["limitations"]
                ),
                created_at=str(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCorruptionError(
                "stored evidence record has an invalid structure"
            ) from exc

    def latest_project(
        self,
        project_id: str,
    ) -> ProjectRecord | None:
        """Return the newest verified state for one project."""

        clean_project_id = project_id.strip()
        if not clean_project_id:
            raise ValueError("project_id must not be empty")

        self.verify_integrity()
        for envelope in reversed(self.load_chain()):
            if envelope.record_type not in {
                "project",
                "project_state",
            }:
                continue

            payload = dict(envelope.payload)
            if payload.get("project_id") == clean_project_id:
                return self._project_from_payload(payload)

        return None

    def save_project(
        self,
        project: ProjectRecord,
    ) -> AuditEnvelope:
        """Save initial project identity or append a newer project state."""

        existing = self.latest_project(project.project_id)

        if existing is None:
            return self.append("project", project.to_payload())

        immutable_fields = (
            "name",
            "repository",
            "worktree",
            "creator",
            "designation",
            "development_track",
        )
        changed = [
            field_name
            for field_name in immutable_fields
            if getattr(existing, field_name) != getattr(project, field_name)
        ]
        if changed:
            raise StoreReferenceError(
                "project identity changed unexpectedly: "
                + ", ".join(changed)
            )

        return self.append("project_state", project.to_payload())

    def approval_question(
        self,
        question_id: str,
    ) -> ApprovalQuestion | None:
        """Return one verified approval question by its identity."""

        clean_question_id = question_id.strip()
        if not clean_question_id:
            raise ValueError("question_id must not be empty")

        self.verify_integrity()
        for envelope in reversed(self.load_chain()):
            if envelope.record_type != "approval_question":
                continue

            payload = dict(envelope.payload)
            if payload.get("question_id") == clean_question_id:
                return self._question_from_payload(payload)

        return None

    def save_approval_question(
        self,
        question: ApprovalQuestion,
    ) -> AuditEnvelope:
        """Store one exact question after verifying its project identity."""

        if self.approval_question(question.question_id) is not None:
            raise StoreReferenceError(
                "approval question identity already exists"
            )

        project = self.latest_project(question.project_id)
        if project is None:
            raise StoreReferenceError(
                "approval question refers to an unknown project"
            )

        mismatches: list[str] = []
        if question.repository != project.repository:
            mismatches.append("repository")
        if question.branch != project.branch:
            mismatches.append("branch")
        if (
            question.commit_or_artifact_hash
            != project.artifact_hash
        ):
            mismatches.append("commit_or_artifact_hash")

        if question.request_kind == "transition":
            if question.scope != project.approved_files:
                mismatches.append("scope")

        if mismatches:
            raise StoreReferenceError(
                "approval question does not match project: "
                + ", ".join(mismatches)
            )

        return self.append(
            "approval_question",
            question.identity_payload(),
        )

    def authorization_grant(
        self,
        grant_id: str,
    ) -> AuthorizationGrant | None:
        """Return the newest verified state of one authorization."""

        clean_grant_id = grant_id.strip()
        if not clean_grant_id:
            raise ValueError("grant_id must not be empty")

        self.verify_integrity()
        for envelope in reversed(self.load_chain()):
            if envelope.record_type not in {
                "authorization_grant",
                "authorization_state",
            }:
                continue

            payload = dict(envelope.payload)
            if payload.get("grant_id") == clean_grant_id:
                return self._grant_from_payload(payload)

        return None

    def save_authorization_grant(
        self,
        grant: AuthorizationGrant,
    ) -> AuditEnvelope:
        """Store a yes-or-no decision only when it matches its question."""

        if self.authorization_grant(grant.grant_id) is not None:
            raise StoreReferenceError(
                "authorization identity already exists"
            )

        question = self.approval_question(grant.question_id)
        if question is None:
            raise StoreReferenceError(
                "authorization refers to an unknown question"
            )

        expected = {
            "project_id": question.project_id,
            "question_text_hash": question.text_hash,
            "grant_type": question.request_kind,
            "requested_value": question.requested_value,
            "target": question.target,
            "scope": question.scope,
            "repository": question.repository,
            "branch": question.branch,
            "commit_or_artifact_hash": (
                question.commit_or_artifact_hash
            ),
            "environment": question.environment,
            "approver": CREATOR,
        }

        mismatches = [
            field_name
            for field_name, expected_value in expected.items()
            if getattr(grant, field_name) != expected_value
        ]
        if mismatches:
            raise StoreReferenceError(
                "authorization does not match question: "
                + ", ".join(mismatches)
            )

        return self.append(
            "authorization_grant",
            grant.to_payload(),
        )

    def evidence_record(
        self,
        evidence_id: str,
    ) -> EvidenceRecord | None:
        """Return one verified evidence record by its identity."""

        clean_evidence_id = evidence_id.strip()
        if not clean_evidence_id:
            raise ValueError("evidence_id must not be empty")

        self.verify_integrity()
        for envelope in reversed(self.load_chain()):
            if envelope.record_type != "evidence":
                continue

            payload = dict(envelope.payload)
            if payload.get("evidence_id") == clean_evidence_id:
                return self._evidence_from_payload(payload)

        return None

    def save_evidence(
        self,
        evidence: EvidenceRecord,
    ) -> AuditEnvelope:
        """Store evidence only for the exact current project artifact."""

        if self.evidence_record(evidence.evidence_id) is not None:
            raise StoreReferenceError(
                "evidence identity already exists"
            )

        project = self.latest_project(evidence.project_id)
        if project is None:
            raise StoreReferenceError(
                "evidence refers to an unknown project"
            )

        if evidence.stage is not project.current_stage:
            raise StoreReferenceError(
                "evidence stage does not match the current project stage"
            )

        try:
            validate_evidence(project, evidence)
        except Exception as exc:
            raise StoreReferenceError(
                f"evidence does not match the governed artifact: {exc}"
            ) from exc

        return self.append("evidence", evidence.to_payload())

    def questions_for_project(
        self,
        project_id: str,
    ) -> tuple[ApprovalQuestion, ...]:
        """Return all verified questions belonging to one project."""

        self.verify_integrity()
        return tuple(
            self._question_from_payload(dict(envelope.payload))
            for envelope in self.load_chain()
            if envelope.record_type == "approval_question"
            and envelope.payload.get("project_id") == project_id
        )

    def grants_for_project(
        self,
        project_id: str,
    ) -> tuple[AuthorizationGrant, ...]:
        """Return all verified yes-or-no records for one project."""

        self.verify_integrity()
        return tuple(
            self._grant_from_payload(dict(envelope.payload))
            for envelope in self.load_chain()
            if envelope.record_type == "authorization_grant"
            and envelope.payload.get("project_id") == project_id
        )

    def evidence_for_project(
        self,
        project_id: str,
    ) -> tuple[EvidenceRecord, ...]:
        """Return all verified evidence belonging to one project."""

        self.verify_integrity()
        return tuple(
            self._evidence_from_payload(dict(envelope.payload))
            for envelope in self.load_chain()
            if envelope.record_type == "evidence"
            and envelope.payload.get("project_id") == project_id
        )


    def append_batch(
        self,
        records: tuple[tuple[str, Mapping[str, Any]], ...],
        *,
        expected_head: str | None = None,
    ) -> tuple[AuditEnvelope, ...]:
        """Append related records together or preserve none of them."""

        if not records:
            raise ValueError("at least one audit record is required")

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.verify_integrity()

            current_head = self._metadata_value("chain_head")
            current_length = int(
                self._metadata_value("chain_length")
            )

            if (
                expected_head is not None
                and current_head != expected_head
            ):
                raise StoreConflictError(
                    "the audit store changed before this operation "
                    "could be saved"
                )

            previous_hash = current_head
            envelopes: list[AuditEnvelope] = []

            for record_type, payload in records:
                clean_type = record_type.strip()
                if not clean_type:
                    raise ValueError(
                        "audit record type must not be empty"
                    )

                envelope = AuditEnvelope.seal(
                    clean_type,
                    dict(payload),
                    previous_hash,
                )

                self.connection.execute(
                    """
                    INSERT INTO audit_records(
                        record_type,
                        payload_json,
                        previous_record_hash,
                        record_hash,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        envelope.record_type,
                        json.dumps(
                            dict(envelope.payload),
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        envelope.previous_record_hash,
                        envelope.record_hash,
                        utc_now_iso(),
                    ),
                )

                envelopes.append(envelope)
                previous_hash = envelope.record_hash

            self._set_metadata("chain_head", previous_hash)
            self._set_metadata(
                "chain_length",
                str(current_length + len(envelopes)),
            )
            self.connection.commit()
            return tuple(envelopes)
        except Exception:
            self.connection.rollback()
            raise

    def current_grants_for_project(
        self,
        project_id: str,
    ) -> tuple[AuthorizationGrant, ...]:
        """Return only the newest state of every project authorization."""

        clean_project_id = project_id.strip()
        if not clean_project_id:
            raise ValueError("project_id must not be empty")

        self.verify_integrity()
        latest: dict[str, AuthorizationGrant] = {}

        for envelope in self.load_chain():
            if envelope.record_type not in {
                "authorization_grant",
                "authorization_state",
            }:
                continue

            payload = dict(envelope.payload)
            if payload.get("project_id") != clean_project_id:
                continue

            grant = self._grant_from_payload(payload)
            latest[grant.grant_id] = grant

        return tuple(latest.values())

    def apply_stage_transition(
        self,
        project_id: str,
        destination: Stage,
        question_id: str,
        grant_id: str,
    ) -> tuple[ProjectRecord, AuthorizationGrant]:
        """Consume one exact approval and move its project atomically."""

        project = self.latest_project(project_id)
        if project is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown project"
            )

        question = self.approval_question(question_id)
        if question is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown question"
            )

        grant = self.authorization_grant(grant_id)
        if grant is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown authorization"
            )

        expected_head = self.chain_head

        updated_project, consumed_grant = apply_transition(
            project,
            destination,
            question,
            grant,
        )

        self.append_batch(
            (
                (
                    "authorization_state",
                    consumed_grant.to_payload(),
                ),
                (
                    "project_state",
                    updated_project.to_payload(),
                ),
            ),
            expected_head=expected_head,
        )

        return updated_project, consumed_grant


# GARVIS STAGE-GATE PROTOTYPE PART 2C COMPLETE


    def append_batch(
        self,
        records: tuple[tuple[str, Mapping[str, Any]], ...],
        *,
        expected_head: str | None = None,
    ) -> tuple[AuditEnvelope, ...]:
        """Append related records together or preserve none of them."""

        if not records:
            raise ValueError("at least one audit record is required")

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.verify_integrity()

            current_head = self._metadata_value("chain_head")
            current_length = int(
                self._metadata_value("chain_length")
            )

            if (
                expected_head is not None
                and current_head != expected_head
            ):
                raise StoreConflictError(
                    "the audit store changed before this operation "
                    "could be saved"
                )

            previous_hash = current_head
            envelopes: list[AuditEnvelope] = []

            for record_type, payload in records:
                clean_type = record_type.strip()
                if not clean_type:
                    raise ValueError(
                        "audit record type must not be empty"
                    )

                envelope = AuditEnvelope.seal(
                    clean_type,
                    dict(payload),
                    previous_hash,
                )

                self.connection.execute(
                    """
                    INSERT INTO audit_records(
                        record_type,
                        payload_json,
                        previous_record_hash,
                        record_hash,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        envelope.record_type,
                        json.dumps(
                            dict(envelope.payload),
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        envelope.previous_record_hash,
                        envelope.record_hash,
                        utc_now_iso(),
                    ),
                )

                envelopes.append(envelope)
                previous_hash = envelope.record_hash

            self._set_metadata("chain_head", previous_hash)
            self._set_metadata(
                "chain_length",
                str(current_length + len(envelopes)),
            )
            self.connection.commit()
            return tuple(envelopes)
        except Exception:
            self.connection.rollback()
            raise

    def current_grants_for_project(
        self,
        project_id: str,
    ) -> tuple[AuthorizationGrant, ...]:
        """Return only the newest state of every project authorization."""

        clean_project_id = project_id.strip()
        if not clean_project_id:
            raise ValueError("project_id must not be empty")

        self.verify_integrity()
        latest: dict[str, AuthorizationGrant] = {}

        for envelope in self.load_chain():
            if envelope.record_type not in {
                "authorization_grant",
                "authorization_state",
            }:
                continue

            payload = dict(envelope.payload)
            if payload.get("project_id") != clean_project_id:
                continue

            grant = self._grant_from_payload(payload)
            latest[grant.grant_id] = grant

        return tuple(latest.values())

    def apply_stage_transition(
        self,
        project_id: str,
        destination: Stage,
        question_id: str,
        grant_id: str,
    ) -> tuple[ProjectRecord, AuthorizationGrant]:
        """Consume one exact approval and move its project atomically."""

        project = self.latest_project(project_id)
        if project is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown project"
            )

        question = self.approval_question(question_id)
        if question is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown question"
            )

        grant = self.authorization_grant(grant_id)
        if grant is None:
            raise StoreReferenceError(
                "stage transition refers to an unknown authorization"
            )

        expected_head = self.chain_head

        updated_project, consumed_grant = apply_transition(
            project,
            destination,
            question,
            grant,
        )

        self.append_batch(
            (
                (
                    "authorization_state",
                    consumed_grant.to_payload(),
                ),
                (
                    "project_state",
                    updated_project.to_payload(),
                ),
            ),
            expected_head=expected_head,
        )

        return updated_project, consumed_grant


# GARVIS STAGE-GATE PROTOTYPE PART 2C COMPLETE


# GARVIS STAGE-GATE PROTOTYPE PART 2B COMPLETE


# GARVIS STAGE-GATE PROTOTYPE PART 2A COMPLETE
