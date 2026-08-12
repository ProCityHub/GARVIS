"""Heartbeat supervisor: council advisory report and supervisor stub."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    request_sha256: str
    consultation_available: bool = True
    council_participation_count: int = 0
    angel_participation_count: int = 0
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Consults the agent council before protected actions."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        digest = hashlib.sha256(message.encode()).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=digest,
            consultation_available=True,
            council_participation_count=0,
            angel_participation_count=0,
            operational_authorization=False,
        )
