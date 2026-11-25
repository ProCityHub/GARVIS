"""Evidence verifier v1.

Methods are file exists, substring match, and SHA-256 match. If the evidence
cannot be opened, the result is ``cannot determine``. There is no truth score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

CANNOT_DETERMINE = "cannot determine"


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    method: str
    checked_at: str
    summary: str
    evidence_ptr: str | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _undetermined(evidence_ptr: str | None) -> VerificationResult:
    return VerificationResult(
        ok=False,
        method="unopened",
        checked_at=_now(),
        summary=CANNOT_DETERMINE,
        evidence_ptr=evidence_ptr,
        reason=CANNOT_DETERMINE,
    )


def _resolve(evidence_ptr: str, root: Path) -> Path | None:
    raw = Path(evidence_ptr)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def verify_claim(
    claim: str,
    evidence_ptr: str | None,
    *,
    root: Path,
    expect_substring: str | None = None,
    expect_sha256: str | None = None,
) -> VerificationResult:
    """Open ``evidence_ptr`` under ``root`` and check it, or say we cannot."""

    del claim
    if not evidence_ptr:
        return _undetermined(None)
    path = _resolve(evidence_ptr, root)
    if path is None:
        return _undetermined(evidence_ptr)
    try:
        if not path.is_file():
            return _undetermined(evidence_ptr)
        data = path.read_bytes()
    except OSError:
        return _undetermined(evidence_ptr)

    checked_at = _now()
    pointer = str(path)
    if expect_sha256:
        digest = sha256(data).hexdigest()
        ok = digest.lower() == expect_sha256.strip().lower()
        return VerificationResult(
            ok=ok,
            method="sha256",
            checked_at=checked_at,
            summary="sha256 match" if ok else "sha256 mismatch",
            evidence_ptr=pointer,
            reason="sha256 match" if ok else "sha256 mismatch",
        )
    if expect_substring is not None:
        text = data.decode("utf-8", errors="replace")
        ok = expect_substring in text
        return VerificationResult(
            ok=ok,
            method="substring",
            checked_at=checked_at,
            summary="substring match" if ok else "substring mismatch",
            evidence_ptr=pointer,
            reason="substring match" if ok else "substring mismatch",
        )
    return VerificationResult(
        ok=True,
        method="file_exists",
        checked_at=checked_at,
        summary="file exists",
        evidence_ptr=pointer,
        reason="file exists",
    )
