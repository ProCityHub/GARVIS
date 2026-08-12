"""Council heartbeat consultation primitives for protected runtime actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Bounded council consultation summary."""

    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Minimal fail-safe supervisor used by capability runtime."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = Path(repository_root)

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=digest,
            consultation_available=True,
            council_participation_count=1 if protected_action else 0,
            angel_participation_count=1 if protected_action else 0,
            operational_authorization=False,
        )
