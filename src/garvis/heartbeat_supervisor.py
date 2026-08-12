"""Heartbeat supervisor: council consultation and advisory reporting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    """Immutable advisory report returned by a council consultation."""

    request_sha256: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    operational_authorization: bool = False


class FullAgentHeartbeatSupervisor:
    """Supervisor that performs council consultations for capability gating.

    In the current implementation the council is always treated as
    available (fail-open for non-protected actions, fail-soft for protected
    ones).  The :class:`CapabilityAwareRuntime` is responsible for
    interpreting the report and deciding whether to proceed.
    """

    def __init__(self, repository_root: Path) -> None:
        # repository_root is retained for future sub-class use or inspection
        self.repository_root = Path(repository_root)

    def consult(
        self,
        message: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        """Return an advisory report for *message*.

        Parameters
        ----------
        message:
            The request text being evaluated.
        protected_action:
            ``True`` when the consultation guards a protected operation such
            as internet research or local-file access.
        """
        digest = hashlib.sha256(message.encode()).hexdigest()
        # operational_authorization is always False: the supervisor reports
        # availability but never grants operational authority directly.  The
        # CapabilityAwareRuntime interprets consultation_available to decide
        # whether a protected action may proceed.
        return CouncilAdvisoryReport(
            request_sha256=digest,
            consultation_available=True,
            council_participation_count=1,
            angel_participation_count=0,
            operational_authorization=False,
        )
