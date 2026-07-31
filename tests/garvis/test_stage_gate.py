from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from garvis.stage_gate import (
    AuditChainError,
    AuthorizationConsumedError,
    AuthorizationDeniedError,
    AuthorizationExpiredError,
    AuthorizationMismatchError,
    AuthorizationRevokedError,
    EvidenceMismatchError,
    InvalidTransitionError,
    ProjectRecord,
    ProtectedAction,
    Stage,
    apply_transition,
    capability_claim_status,
    create_evidence,
    create_protected_action_question,
    create_transition_question,
    grant_from_answer,
    new_identifier,
    render_project_status,
    require_legal_transition,
    revoke_grant,
    seal_audit_records,
    validate_evidence,
    validate_grant,
    verify_audit_chain,
)
from garvis.stage_gate_store import (
    StageGateStore,
    StoreConflictError,
    StoreCorruptionError,
    StoreReferenceError,
)

APPROVED_FILES = (
    "src/garvis/stage_gate.py",
    "src/garvis/stage_gate_store.py",
    "tests/garvis/test_stage_gate.py",
    "docs/GARVIS_STAGE_GATE_FOUNDATION.md",
)


def make_project(
    *,
    stage: Stage = Stage.PROTOTYPE,
    artifact_hash: str = "approved-prototype-hash",
) -> ProjectRecord:
    return ProjectRecord(
        project_id=new_identifier("project"),
        name="GARVIS AGI Beta Stage-Gate Foundation",
        repository="ProCityHub/GARVIS",
        worktree="~/GARVIS-stage-gate",
        branch="feature/stage-gate-foundation-v1",
        base_commit="e88b9d4",
        artifact_hash=artifact_hash,
        approved_files=APPROVED_FILES,
        current_stage=stage,
        pending_gate=f"{stage.value} -> next approved stage",
    )


def test_capability_claim_remains_honest_agi_beta() -> None:
    status = capability_claim_status(
        scientifically_validated=False,
    )

    assert status["designation"] == "GARVIS AGI Beta"
    assert status["development_track"] == "UPGRADE 2"
    assert status["scientifically_validated_agi"] is False
    assert (
        status["scientific_validation_status"]
        == "Full AGI is a development objective; scientific validation is not established."
    )


def test_only_legal_governance_transitions_are_accepted() -> None:
    require_legal_transition(
        Stage.SPECIFICATION,
        Stage.PROTOTYPE,
    )

    with pytest.raises(InvalidTransitionError):
        require_legal_transition(
            Stage.SPECIFICATION,
            Stage.MERGE,
        )


def test_transition_consumes_one_time_approval() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize local implementation inside the approved boundary.",
    )
    grant = grant_from_answer(question, "yes")

    updated, consumed = apply_transition(
        project,
        Stage.PROTOTYPE,
        question,
        grant,
    )

    assert updated.current_stage is Stage.PROTOTYPE
    assert consumed.consumed_at is not None

    with pytest.raises(AuthorizationConsumedError):
        validate_grant(question, consumed)


def test_changed_question_cannot_reuse_previous_yes() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize the exact approved Prototype.",
    )
    grant = grant_from_answer(question, "yes")

    changed_question = replace(
        question,
        explanation="This wording changes the approval context.",
    )

    with pytest.raises(AuthorizationMismatchError):
        validate_grant(changed_question, grant)


def test_direct_no_cannot_authorize_protected_action() -> None:
    project = make_project()
    question = create_protected_action_question(
        project,
        ProtectedAction.PUSH,
        "GitHub branch feature/stage-gate-foundation-v1",
        "Upload one exact approved commit without merging it.",
    )
    denial = grant_from_answer(question, "no")

    with pytest.raises(AuthorizationDeniedError):
        validate_grant(question, denial)


def test_expired_approval_is_rejected() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize the approved local Prototype.",
    )
    expired_question = replace(
        question,
        expires_at=(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat().replace("+00:00", "Z"),
    )
    grant = grant_from_answer(expired_question, "yes")

    with pytest.raises(AuthorizationExpiredError):
        validate_grant(expired_question, grant)


def test_revoked_approval_is_rejected() -> None:
    project = make_project()
    question = create_protected_action_question(
        project,
        ProtectedAction.PUSH,
        "GitHub branch feature/stage-gate-foundation-v1",
        "Upload one exact approved commit.",
    )
    grant = grant_from_answer(question, "yes")
    revoked = revoke_grant(
        grant,
        "Adrien withdrew the unused approval.",
    )

    with pytest.raises(AuthorizationRevokedError):
        validate_grant(question, revoked)


def test_evidence_is_bound_to_exact_artifact() -> None:
    project = make_project()
    evidence = create_evidence(
        project,
        "prototype_smoke",
        "pass",
        "Local Prototype verification.",
    )

    validate_evidence(project, evidence)

    changed_evidence = replace(
        evidence,
        artifact_hash="different-artifact-hash",
    )

    with pytest.raises(EvidenceMismatchError):
        validate_evidence(project, changed_evidence)


def test_audit_chain_allows_references_but_blocks_duplicate_records() -> None:
    project = make_project()
    question = create_protected_action_question(
        project,
        ProtectedAction.PUSH,
        "GitHub branch feature/stage-gate-foundation-v1",
        "Upload one exact approved commit.",
    )
    grant = grant_from_answer(question, "yes")

    chain = seal_audit_records(
        (
            ("project", project.to_payload()),
            (
                "approval_question",
                question.identity_payload(),
            ),
            (
                "authorization_grant",
                grant.to_payload(),
            ),
        )
    )

    assert verify_audit_chain(chain)

    duplicated_question_chain = seal_audit_records(
        (
            (
                "approval_question",
                question.identity_payload(),
            ),
            (
                "approval_question",
                question.identity_payload(),
            ),
        )
    )

    with pytest.raises(AuditChainError):
        verify_audit_chain(duplicated_question_chain)


def test_audit_tampering_and_status_limitations_are_visible() -> None:
    project = make_project()
    evidence = create_evidence(
        project,
        "prototype_smoke",
        "pass",
        "Local Prototype verification.",
    )

    chain = seal_audit_records(
        (
            ("project", project.to_payload()),
            ("evidence", evidence.to_payload()),
        )
    )

    tampered_chain = (
        chain[0],
        replace(
            chain[1],
            previous_record_hash="tampered",
        ),
    )

    with pytest.raises(AuditChainError):
        verify_audit_chain(tampered_chain)

    brief = render_project_status(project)
    detailed = render_project_status(
        project,
        evidence=(evidence,),
        detailed=True,
    )

    assert "GARVIS AGI Beta" in brief
    assert "scientific validation is not established" in brief
    assert "Current stage: prototype" in brief
    assert "Push or force-push" in detailed
    assert "Deployment or rollback" in detailed



# Storage and recovery tests


def test_store_reopens_typed_governance_records() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize the exact approved local Prototype.",
    )
    grant = grant_from_answer(question, "yes")
    evidence = create_evidence(
        project,
        "specification_integrity",
        "pass",
        "Verified the exact governing specification.",
    )

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "typed-records.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            store.save_approval_question(question)
            store.save_authorization_grant(grant)
            store.save_evidence(evidence)
            assert store.verify_integrity()

        with StageGateStore(database) as reopened:
            assert reopened.latest_project(
                project.project_id
            ) == project
            assert reopened.approval_question(
                question.question_id
            ) == question
            assert reopened.authorization_grant(
                grant.grant_id
            ) == grant
            assert reopened.evidence_record(
                evidence.evidence_id
            ) == evidence
            assert reopened.verify_integrity()


def test_store_rejects_authorization_for_wrong_branch() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize the exact approved local Prototype.",
    )
    grant = grant_from_answer(question, "yes")
    mismatched = replace(
        grant,
        grant_id=new_identifier("grant"),
        branch="wrong-branch",
    )

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "wrong-branch.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            store.save_approval_question(question)
            before_count = store.record_count

            with pytest.raises(StoreReferenceError):
                store.save_authorization_grant(mismatched)

            assert store.record_count == before_count
            assert store.verify_integrity()


def test_store_applies_transition_and_consumes_grant_atomically() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    question = create_transition_question(
        project,
        Stage.PROTOTYPE,
        "Authorize the approved Prototype transition.",
    )
    grant = grant_from_answer(question, "yes")

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "transition.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            store.save_approval_question(question)
            store.save_authorization_grant(grant)

            before_count = store.record_count
            updated, consumed = store.apply_stage_transition(
                project.project_id,
                Stage.PROTOTYPE,
                question.question_id,
                grant.grant_id,
            )

            assert store.record_count == before_count + 2
            assert updated.current_stage is Stage.PROTOTYPE
            assert consumed.consumed_at is not None

        with StageGateStore(database) as reopened:
            restored_project = reopened.latest_project(
                project.project_id
            )
            restored_grant = reopened.authorization_grant(
                grant.grant_id
            )

            assert restored_project is not None
            assert (
                restored_project.current_stage
                is Stage.PROTOTYPE
            )
            assert restored_grant is not None
            assert restored_grant.consumed_at is not None
            assert reopened.verify_integrity()


def test_store_rolls_back_failed_multi_record_append() -> None:
    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "rollback.db"

        with StageGateStore(database) as store:
            before_count = store.record_count
            before_head = store.chain_head

            with pytest.raises(TypeError):
                store.append_batch(
                    (
                        (
                            "temporary_valid_record",
                            {"status": "must not remain"},
                        ),
                        (
                            "temporary_invalid_record",
                            {"unsupported": {"not", "json"}},
                        ),
                    )
                )

            assert store.record_count == before_count
            assert store.chain_head == before_head
            assert store.verify_integrity()


def test_store_blocks_update_from_stale_chain_head() -> None:
    project = make_project()

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "stale-head.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            stale_head = store.chain_head

            store.append(
                "concurrency_anchor",
                {
                    "project_id": project.project_id,
                    "status": "newer operation",
                },
            )
            current_count = store.record_count

            with pytest.raises(StoreConflictError):
                store.append_batch(
                    (
                        (
                            "stale_operation",
                            {
                                "project_id": project.project_id,
                                "status": "must be refused",
                            },
                        ),
                    ),
                    expected_head=stale_head,
                )

            assert store.record_count == current_count
            assert store.verify_integrity()


def test_store_detects_modified_database_record() -> None:
    project = make_project()

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "tampered.db"

        with StageGateStore(database) as store:
            store.save_project(project)

        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                UPDATE audit_records
                SET payload_json = ?
                WHERE sequence = 1
                """,
                ('{"tampered":true}',),
            )
            connection.commit()

        with StageGateStore(database) as tampered:
            with pytest.raises(StoreCorruptionError):
                tampered.verify_integrity()


def test_store_detects_deleted_final_record() -> None:
    project = make_project()

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "deleted-tail.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            store.append(
                "tail_record",
                {
                    "project_id": project.project_id,
                    "status": "must remain",
                },
            )
            assert store.record_count == 2

        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                DELETE FROM audit_records
                WHERE sequence = (
                    SELECT MAX(sequence) FROM audit_records
                )
                """
            )
            connection.commit()

        with StageGateStore(database) as shortened:
            with pytest.raises(StoreCorruptionError):
                shortened.verify_integrity()


def test_store_preserves_project_history_and_returns_latest_state() -> None:
    project = make_project(stage=Stage.SPECIFICATION)
    updated = replace(
        project,
        current_stage=Stage.PROTOTYPE,
        pending_gate="prototype -> tests",
    )

    with TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "history.db"

        with StageGateStore(database) as store:
            store.save_project(project)
            store.save_project(updated)

            latest = store.latest_project(project.project_id)
            chain = store.load_chain()

            assert latest == updated
            assert len(chain) == 2
            assert chain[0].record_type == "project"
            assert chain[1].record_type == "project_state"
            assert store.verify_integrity()


# GARVIS STAGE-GATE PROTOTYPE PART 3C COMPLETE

# GARVIS STAGE-GATE PROTOTYPE PART 3A COMPLETE
