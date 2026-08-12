"""Council advisory heartbeat supervisor for GARVIS capability runtime."""

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
    """Default heartbeat supervisor that consults the local council state."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = Path(repository_root)

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        """Consult the council and return an advisory report.

        The default implementation always marks consultation as available and
        does not grant operational authorization.  Callers that require richer
        governance behaviour should inject a custom supervisor.
        """
        digest = hashlib.sha256(message.encode()).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=digest,
            consultation_available=True,
            council_participation_count=0,
            angel_participation_count=0,
            operational_authorization=False,
        )
