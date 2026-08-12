"""Heartbeat supervisor providing council advisory reports for GARVIS."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Advisory report returned by a heartbeat supervisor consultation."""

    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool


class FullAgentHeartbeatSupervisor:
    """Heartbeat supervisor that records council consultations."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        request_sha256 = hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=request_sha256,
            consultation_available=True,
            council_participation_count=0,
            angel_participation_count=0,
            operational_authorization=False,
        )
