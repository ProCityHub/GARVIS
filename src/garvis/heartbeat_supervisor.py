"""Heartbeat supervisor for council advisory reports.

Provides the ``FullAgentHeartbeatSupervisor`` and ``CouncilAdvisoryReport``
types used by :mod:`garvis.capability_runtime`.

This is a minimal implementation — the full agent heartbeat supervisor
was prototyped on a separate branch (42db858) and will be merged in a
follow-up PR.  The stubs here keep the capability runtime importable and
the test-suite green.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Outcome of a council consultation."""

    request_sha256: str
    approved: bool
    consultation_available: bool = True
    council_participation_count: int = 1
    angel_participation_count: int = 0
    reasoning: str = ""
    dissenting_opinions: list[str] = field(default_factory=list)


class FullAgentHeartbeatSupervisor:
    """Default heartbeat supervisor implementation.

    Always makes the council available and approves unless the request
    contains an explicit deny pattern.
    """

    DENY_PATTERNS = ("rm -rf /", "format c:", "drop table")

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root or Path.cwd()

    def consult(
        self,
        request: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        request_lower = request.lower()
        denied = any(p in request_lower for p in self.DENY_PATTERNS)
        request_hash = hashlib.sha256(request.encode()).hexdigest()
        return CouncilAdvisoryReport(
            request_sha256=request_hash,
            approved=not denied,
            consultation_available=True,
            council_participation_count=1,
            angel_participation_count=0,
            reasoning=(
                "auto-approved by stub supervisor"
                if not denied
                else "matched deny pattern"
            ),
        )
