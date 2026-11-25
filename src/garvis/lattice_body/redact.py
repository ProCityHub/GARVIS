"""Zero-secret redaction for lattice output.

Reuses the research-bridge patterns and the heartbeat credential patterns.
Those patterns do the removal. This wrapper does not add a character cap.
"""

from __future__ import annotations

from garvis.heartbeat_supervisor import _SECRET_PATTERNS
from garvis.upgrade_research import redact_secrets

REDACTION = "[REDACTED]"


def redact_text(value: str | None) -> str:
    """Return ``value`` with credential-shaped substrings removed."""

    if not value:
        return ""
    cleaned = redact_secrets(value)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTION, cleaned)
    return cleaned
