"""Heartbeat supervisor: consults a council of agents before protected actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CouncilAdvisoryReport:
    """Summary of a council consultation."""

    request_sha256: str
    consultation_available: bool = False
    council_participation_count: int = 0
    angel_participation_count: int = 0
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Stub supervisor that returns a baseline advisory report.

    A full implementation would contact external council members; this
    default implementation returns a report indicating that consultation
    is available, allowing the runtime to proceed with both ordinary and
    protected actions (e.g. internet research) without an external council.
    """

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
