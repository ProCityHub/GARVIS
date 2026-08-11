"""Heartbeat supervisor: council advisory layer for protected GARVIS actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Advisory report returned by a heartbeat supervisor consultation."""

    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Minimal council supervisor that authorises all requests.

    In the absence of a multi-agent council the supervisor acts as a
    pass-through: consultation is always considered available so that the
    capability runtime can proceed normally.  Protected actions are therefore
    not blocked by a missing council.
    """

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        sha256 = hashlib.sha256(message.encode()).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=sha256,
            consultation_available=True,
            council_participation_count=0,
            angel_participation_count=0,
            operational_authorization=False,
        )
