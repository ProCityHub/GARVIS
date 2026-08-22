import math

import pytest

from garvis.heartbeat_kernel import (
    ALPHA,
    BETA,
    ClaimStatus,
    benchmark_phi,
)
from garvis.heartbeat_service import AutomaticHeartbeatService
from garvis.creator_authority import CreatorAuthority
from garvis.self_authority import GarvisSelfAuthority, InternalAction


def test_phi_identity() -> None:
    assert math.isclose(
        ALPHA + BETA,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_heartbeat_advances_persists_and_self_claims(tmp_path) -> None:
    service = AutomaticHeartbeatService(
        tmp_path,
        interval_seconds=0.0,
    )
    try:
        state = service.run_once()
        assert state.status.value == "completed"
        assert state.self_claims
        assert all(
            claim.status is ClaimStatus.SELF_CLAIM
            for claim in state.self_claims
        )
        health = service.health()
        assert health["sequence"] == 1
        assert health["heartbeat_running"] is True
        assert health["creator"] == "Adrien D. Thomas"
    finally:
        service.close()

    restarted = AutomaticHeartbeatService(
        tmp_path,
        interval_seconds=0.0,
    )
    try:
        assert restarted.sequence == 1
        restarted.run_once()
        assert restarted.sequence == 2
    finally:
        restarted.close()


def test_phi_is_benchmarked_not_privileged() -> None:
    result = benchmark_phi(
        observer=1.0,
        actor=0.2,
        bridge=0.9,
        target=0.5,
    )
    assert result["status"] == "HYPOTHESIS_UNDER_TEST"
    assert result["winner"]["family"] in {"phi", "lambda"}
    assert len(result["comparisons"]) == 21


def test_real_heartbeat_runs_on_garvis_self_authority(tmp_path) -> None:
    service = AutomaticHeartbeatService(
        tmp_path,
        interval_seconds=0.0,
        creator_authority=CreatorAuthority(enabled=False),
        self_authority=GarvisSelfAuthority(),
    )
    try:
        state = service.run_once()
        assert state.status.value == "completed"
        assert service.sequence == 1
        assert service.self_authority.permits(InternalAction.HEARTBEAT)
        assert service.self_authority.permits(InternalAction.CONSOLIDATE)
        assert service.self_authority.permits(InternalAction.LEARN)
    finally:
        service.close()


def test_disabled_self_authority_blocks_internal_heartbeat(tmp_path) -> None:
    with pytest.raises(
        PermissionError,
        match="GARVIS self-authority does not permit action: heartbeat",
    ):
        AutomaticHeartbeatService(
            tmp_path,
            interval_seconds=0.0,
            creator_authority=CreatorAuthority(),
            self_authority=GarvisSelfAuthority(enabled=False),
        )
