"""Tests for the ``garvis thanos`` command-line interface."""

from __future__ import annotations

import pytest

from garvis.thanos_cli import main


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GARVIS_HOME", str(tmp_path))
    return tmp_path


def test_enable_reports_standing_authority(home, capsys) -> None:
    assert main(["enable"]) == 0
    out = capsys.readouterr().out
    assert "THANOS_MODE=ENABLED" in out
    assert "OWNER=Adrien D. Thomas" in out
    assert "PER_STAGE_APPROVAL_PROMPTS=0" in out
    assert "OWNER_MERGE_CHECKPOINTS_PER_CYCLE=0" in out
    assert "AUTONOMOUS_MERGE_WHEN_GREEN=ENABLED" in out
    assert "TARGET_VERSION=2.0.0-beta.1" in out


def test_enable_does_not_claim_unbuilt_subsystems(home, capsys) -> None:
    main(["enable"])
    out = capsys.readouterr().out
    assert "ROLLBACK=NOT_IMPLEMENTED" in out
    assert "REPAIR_ENGINE=NOT_IMPLEMENTED" in out
    assert "CAPABILITY_REGISTRY=NOT_IMPLEMENTED" in out


def test_enable_is_idempotent_refusal(home, capsys) -> None:
    main(["enable"])
    capsys.readouterr()
    assert main(["enable"]) == 1
    assert "REFUSED=ALREADY_ENABLED" in capsys.readouterr().out


def test_status_survives_a_restart(home, capsys) -> None:
    main(["enable"])
    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "THANOS_MODE=ENABLED" in out
    assert "ACTIVE_CYCLE=NONE" in out


def test_pause_and_resume_round_trip(home, capsys) -> None:
    main(["enable"])
    capsys.readouterr()
    main(["pause"])
    assert "THANOS_MODE=PAUSED" in capsys.readouterr().out
    main(["resume"])
    assert "THANOS_MODE=ENABLED" in capsys.readouterr().out


def test_revoke_requires_a_reason(home) -> None:
    main(["enable"])
    with pytest.raises(SystemExit):
        main(["revoke"])


def test_revoke_then_reenable_preserves_history(home, capsys) -> None:
    main(["enable"])
    main(["revoke", "--reason", "owner stop"])
    capsys.readouterr()
    assert main(["enable"]) == 0
    assert "SUPERSEDING_REVOKED_AUTHORIZATION" in capsys.readouterr().out

    assert main(["history"]) == 0
    out = capsys.readouterr().out
    assert "REVOKED" in out
    assert "CHAIN_LENGTH=3" in out


def test_revocation_is_not_a_permanent_lockout(home, capsys) -> None:
    main(["enable"])
    main(["revoke", "--reason", "owner stop"])
    main(["enable"])
    capsys.readouterr()
    assert main(["status"]) == 0
    assert "THANOS_MODE=ENABLED" in capsys.readouterr().out


def test_run_reports_not_implemented_rather_than_success(home, capsys) -> None:
    main(["enable"])
    capsys.readouterr()
    assert main(["run"]) == 3
    out = capsys.readouterr().out
    assert "AUTONOMOUS_REPAIR_LOOP=NOT_IMPLEMENTED" in out
    assert "PASS" not in out


def test_health_reports_not_implemented(home, capsys) -> None:
    assert main(["health"]) == 3
    assert "RUNTIME_HEALTH_CHECK=NOT_IMPLEMENTED" in capsys.readouterr().out


def test_pause_without_authorization_is_refused(home, capsys) -> None:
    assert main(["pause"]) == 1
    assert "REFUSED=NO_AUTHORIZATION" in capsys.readouterr().out


def test_tampered_store_is_reported(home, capsys) -> None:
    main(["enable"])
    capsys.readouterr()
    path = home / "thanos.json"
    path.write_text(path.read_text().replace("Adrien D. Thomas", "Someone Else"))
    main(["status"])
    assert "THANOS_STATE=TAMPERED" in capsys.readouterr().out


def test_empty_history(home, capsys) -> None:
    assert main(["history"]) == 0
    assert "AUTHORIZATION_CHAIN=EMPTY" in capsys.readouterr().out
