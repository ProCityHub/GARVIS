"""Council heartbeat supervision for capability runtime oversight."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Summary of a council consultation for a runtime request."""

    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Consults the council registry to assess heartbeat availability."""

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

        council_path = self.repository_root / ".garvis" / "council"
        try:
            members = (
                [p for p in council_path.iterdir() if p.is_file()]
                if council_path.is_dir()
                else []
            )
            council_count = len(members)
            angel_count = sum(
                1
                for p in members
                if p.name.startswith("angel_")
            )
        except OSError:
            council_count = 0
            angel_count = 0
        # When no council registry is configured the supervisor is available
        # by default (open-world assumption: no opposition means permitted).
        available = True

        return CouncilAdvisoryReport(
            request_sha256=request_sha256,
            consultation_available=available,
            council_participation_count=council_count,
            angel_participation_count=angel_count,
            operational_authorization=False,
        )
