"""THANOS is legacy-only; heartbeat liveness is creator-owned."""

from __future__ import annotations

import pytest

from garvis.thanos_cli import main


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GARVIS_HOME", str(tmp_path))
    return tmp_path


def test_enable_does_not_create_thanos_authority(home, capsys) -> None:
    assert main(["enable"]) == 0
    out = capsys.readouterr().out
    assert "THANOS=LEGACY_ONLY" in out
    assert "THANOS_OPERATIONAL_AUTHORITY=DISABLED" in out
    assert "AUTHORITY_SOURCE=CREATOR_DIRECTIVE" in out
    assert "CREATOR=Adrien D. Thomas" in out
    assert not (home / "thanos.json").exists()


def test_run_executes_real_heartbeat(home, capsys) -> None:
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    assert "HEARTBEAT_STATUS=COMPLETED" in out
    assert "NOT_IMPLEMENTED" not in out


def test_health_is_real_not_hardcoded(home, capsys) -> None:
    main(["run"])
    capsys.readouterr()
    assert main(["health"]) == 0
    out = capsys.readouterr().out
    assert "heartbeat_running" in out
    assert "RUNTIME_HEALTH_CHECK=NOT_IMPLEMENTED" not in out


def test_legacy_mutation_commands_are_noops(home, capsys) -> None:
    for argv in (
        ["pause"],
        ["resume"],
        ["history"],
        ["revoke", "--reason", "test"],
    ):
        assert main(argv) == 0
        assert (
            "RESULT=NO_THANOS_AUTHORITY_MUTATION"
            in capsys.readouterr().out
        )
    assert not (home / "thanos.json").exists()
