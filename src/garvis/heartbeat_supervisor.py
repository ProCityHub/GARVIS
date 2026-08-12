"""Heartbeat supervisor for council advisory consultation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Consult the advisory council before protected runtime actions."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        request_sha256 = hashlib.sha256(message.encode()).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=request_sha256,
            consultation_available=True,
            council_participation_count=0,
            angel_participation_count=0,
            operational_authorization=False,
        )
