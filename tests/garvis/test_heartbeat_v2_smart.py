from __future__ import annotations

import subprocess
from pathlib import Path

from garvis.heartbeat_kernel import PredictionWitnessLedger
from garvis.heartbeat_service import AutomaticHeartbeatService
from garvis.thanos_mode import (
    DEFAULT_ALLOWED_ACTIONS,
    ThanosAction,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "heartbeat@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Heartbeat Test"],
        check=True,
    )
    (path / "README.md").write_text("heartbeat\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "branch", "-M", "main"],
        check=True,
    )


def test_smart_heartbeat_observes_talks_and_closes_omega_to_alpha(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    service = AutomaticHeartbeatService(
        tmp_path / "state",
        interval_seconds=0.0,
        repository_root=repo,
    )
    try:
        state = service.run_once()
        health = service.health()

        assert state.status.value == "completed"
        assert state.verification["omega_to_alpha_verified"] is True
        assert health["heartbeat_version"] == "v2-smart"
        assert health["current_phase"] == "RECEIVE"
        assert health["alpha_omega_closure"] == "CONSOLIDATE->RECEIVE"
        assert health["witness_capture"] == "automatic_non_blocking"

        dialogue = health["internal_dialogue"]
        assert dialogue["status"] == "INTERNAL_DIALOGUE_NOT_EVIDENCE"
        assert dialogue["observer"]
        assert dialogue["skeptic"]
        assert dialogue["planner"]
        assert dialogue["repair_candidate"] == "no_repair_needed"
    finally:
        service.close()


def test_dirty_worktree_is_seen_but_not_mutated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    target = repo / "README.md"
    target.write_text("heartbeat changed\n", encoding="utf-8")

    service = AutomaticHeartbeatService(
        tmp_path / "state",
        interval_seconds=0.0,
        repository_root=repo,
    )
    try:
        service.run_once()
        health = service.health()
        dialogue = health["internal_dialogue"]

        assert dialogue["repair_candidate"] == "preserve_then_diagnose_worktree"
        assert target.read_text(encoding="utf-8") == "heartbeat changed\n"
    finally:
        service.close()


def test_prediction_witness_public_api_is_capture(tmp_path):
    ledger = PredictionWitnessLedger(tmp_path / "witness.sqlite3")
    try:
        prediction_id, digest = ledger.capture("cycle", {"expected": 1})
        assert prediction_id.startswith("pred_")
        assert len(digest) == 64
    finally:
        ledger.close()


def test_direct_merge_is_not_standing_authority():
    assert ThanosAction.MERGE not in DEFAULT_ALLOWED_ACTIONS
    assert ThanosAction.REQUEST_MERGE in DEFAULT_ALLOWED_ACTIONS


def test_remote_automation_is_observe_by_default():
    root = Path(__file__).resolve().parents[2]

    docs = (root / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in docs
    assert "workflow_run:" not in docs

    fixer = (root / ".github/workflows/ci-fix-bot.yml").read_text(encoding="utf-8")
    assert "contents: read" in fixer
    assert "issues: read" in fixer
    assert "pull-requests: read" in fixer
    assert "github.rest.pulls.create" not in fixer
    assert "git push origin ${branch}" not in fixer

    square = (root / ".github/workflows/town-square.yml").read_text(encoding="utf-8")
    assert "contents: read" in square
    assert "agent_bus.py route" not in square
