from hashlib import sha256

from garvis.provenance_guard import (
    LEGAL_AUTODECISION,
    LICENSE_AUTOMODIFICATION,
    LOCAL_ONLY,
    NETWORK_CAPABILITY,
    SOURCE_HISTORY_REWRITE,
    evaluate_latest_security_report,
    evaluate_report,
    load_evidence_report,
)


def _write(tmp_path, text):
    path = tmp_path / "report.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_capability_boundaries():
    assert LOCAL_ONLY is True
    assert NETWORK_CAPABILITY is False
    assert LEGAL_AUTODECISION is False
    assert LICENSE_AUTOMODIFICATION is False
    assert SOURCE_HISTORY_REWRITE is False


def test_clean_report_reaches_human_legal_gate(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "LICENSE_EVENT_COUNT=1",
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
    )

    report = load_evidence_report(path)
    decision = evaluate_report(report)

    assert report.sha256 == sha256(path.read_bytes()).hexdigest()
    assert decision.decision == "EVIDENCE_ACCEPTED_FOR_REVIEW"
    assert decision.safe_state is True
    assert decision.legal_conclusion == "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
    assert decision.next_gate == "HUMAN_LEGAL_SCOPE_REVIEW"


def test_network_action_blocks_evidence_acceptance(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=TRUE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.safe_state is False
    assert decision.decision == "EVIDENCE_REJECTED_UNSAFE_STATE"
    assert decision.next_gate == "SECURITY_REVIEW_AND_CREATOR_APPROVAL"


def test_missing_security_flag_is_incomplete(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_INCOMPLETE"
    assert "PROTECTED_ACTION_PERFORMED" in decision.missing_security_flags


def test_technical_claim_cannot_become_legal_conclusion(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
                "GIT_DATE_ALONE_PROVES_LEGAL_PRIORITY=TRUE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_CONTRADICTORY"
    assert decision.legal_conclusion == "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
    assert decision.contradictions

def test_conflicting_duplicate_security_flag_must_not_be_accepted(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "NETWORK_OPERATION_PERFORMED=TRUE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision != "EVIDENCE_ACCEPTED_FOR_REVIEW"
    assert decision.safe_state is False
    assert any(
        "NETWORK_OPERATION_PERFORMED" in item
        for item in decision.contradictions
    )


def test_duplicate_identical_security_flags_remain_safe(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_ACCEPTED_FOR_REVIEW"
    assert decision.safe_state is True


def test_true_before_false_security_conflict_must_remain_unsafe(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=TRUE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision != "EVIDENCE_ACCEPTED_FOR_REVIEW"
    assert decision.safe_state is False

def test_duplicate_legal_false_then_true_must_be_contradictory(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
                "GIT_DATE_ALONE_PROVES_LEGAL_PRIORITY=FALSE",
                "GIT_DATE_ALONE_PROVES_LEGAL_PRIORITY=TRUE",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_CONTRADICTORY"
    assert decision.legal_conclusion == "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
    assert any(
        "GIT_DATE_ALONE_PROVES_LEGAL_PRIORITY" in item
        for item in decision.contradictions
    )


def test_duplicate_event_count_valid_then_malformed_must_be_contradictory(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
                "LICENSE_EVENT_COUNT=1",
                "LICENSE_EVENT_COUNT=banana",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_CONTRADICTORY"
    assert any(
        "LICENSE_EVENT_COUNT" in item
        for item in decision.contradictions
    )


def test_duplicate_event_count_valid_then_negative_must_be_contradictory(tmp_path):
    path = _write(
        tmp_path,
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
                "LICENSE_EVENT_COUNT=1",
                "LICENSE_EVENT_COUNT=-1",
            ]
        ),
    )

    decision = evaluate_report(load_evidence_report(path))

    assert decision.decision == "EVIDENCE_CONTRADICTORY"
    assert any(
        "LICENSE_EVENT_COUNT" in item
        for item in decision.contradictions
    )

def test_latest_security_report_ignores_matching_directory(tmp_path):
    report_path = tmp_path / "GARVIS_LICENSE_ORIGIN_TEMPORAL_SCOPE_001.txt"
    report_path.write_text(
        "\n".join(
            [
                "SOURCE_CHANGED=FALSE",
                "GIT_HISTORY_CHANGED=FALSE",
                "NETWORK_OPERATION_PERFORMED=FALSE",
                "PROTECTED_ACTION_PERFORMED=FALSE",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "GARVIS_LICENSE_ORIGIN_TEMPORAL_SCOPE_999.txt").mkdir()

    report, decision = evaluate_latest_security_report(tmp_path)

    assert report.path == report_path.resolve()
    assert decision.decision == "EVIDENCE_ACCEPTED_FOR_REVIEW"
