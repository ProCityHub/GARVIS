"""Research evidence bridge for GARVIS THANOS upgrade cycles.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

``garvis.internet_research.InternetResearchClient`` already reaches the
network safely. This module is the layer above it: it decides which sources
a self-upgrade may rely on, and it records what was actually retrieved.

Three rules are enforced here rather than assumed:

1. **Tier before trust.** Every source is classified PRIMARY (official
   documentation, package indexes, advisory databases), SECONDARY, or
   UNTRUSTED. A patch justified only by UNTRUSTED evidence is refused.
2. **Content binding.** Each evidence record carries the SHA-256 of the bytes
   actually retrieved. A claim is therefore falsifiable later: if the page
   changed, the hash no longer matches and the justification is stale.
3. **No secrets leave the process.** Text is redacted before it is written to
   the ledger, so a token pasted into a URL or echoed by an error never
   reaches durable storage.

Fetched content is evidence, never instructions. Nothing retrieved here is
executed, and nothing here applies a patch.

Python 3.9 compatible. Termux-safe: offline degrades to a recorded blocker
rather than an exception storm.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from garvis.stage_gate import canonical_json, new_identifier, sha256_payload, utc_now_iso

__all__ = [
    "PRIMARY_DOMAINS",
    "SECONDARY_DOMAINS",
    "EvidenceLedger",
    "EvidenceError",
    "InsufficientEvidenceError",
    "ResearchEvidence",
    "ResearchProvider",
    "SourceTier",
    "StaticResearchProvider",
    "classify_source",
    "contains_secret",
    "evidence_from_source",
    "record_all",
    "redact_secrets",
    "sufficient_for_patch",
]

GENESIS_HASH = "0" * 64

#: Official documentation, package metadata, and advisory databases.
PRIMARY_DOMAINS = frozenset(
    {
        "docs.python.org",
        "peps.python.org",
        "packaging.python.org",
        "pypi.org",
        "api.github.com",
        "docs.astral.sh",
        "osv.dev",
        "api.osv.dev",
        "nvd.nist.gov",
        "cve.mitre.org",
        "docs.pytest.org",
        "mypy.readthedocs.io",
        "setuptools.pypa.io",
    }
)

#: Community and repository content: useful for orientation, not sole justification.
SECONDARY_DOMAINS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "stackoverflow.com",
        "wikipedia.org",
        "en.wikipedia.org",
        "readthedocs.io",
        "realpython.com",
    }
)


class SourceTier(str, Enum):
    """Trust tier of a retrieved source."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    UNTRUSTED = "UNTRUSTED"


class EvidenceError(RuntimeError):
    """Base error for the evidence bridge."""


class InsufficientEvidenceError(EvidenceError):
    """Raised when a patch is attempted without adequate source support."""


_SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[=:]\s*\S+"),
    re.compile(r"(?i)://[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

REDACTION = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Return ``text`` with credential-shaped substrings removed."""

    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTION, cleaned)
    return cleaned


def contains_secret(text: str) -> bool:
    """Return True when ``text`` appears to carry credential material."""

    return redact_secrets(text) != text


def _domain_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify_source(url: str) -> SourceTier:
    """Classify ``url`` into a trust tier by its registered domain."""

    domain = _domain_of(url)
    if not domain:
        return SourceTier.UNTRUSTED
    if domain in PRIMARY_DOMAINS:
        return SourceTier.PRIMARY
    if domain in SECONDARY_DOMAINS:
        return SourceTier.SECONDARY

    parts = domain.split(".")
    for index in range(1, len(parts) - 1):
        parent = ".".join(parts[index:])
        if parent in PRIMARY_DOMAINS:
            return SourceTier.PRIMARY
        if parent in SECONDARY_DOMAINS:
            return SourceTier.SECONDARY
    return SourceTier.UNTRUSTED


@dataclass(frozen=True)
class ResearchEvidence:
    """One retrieved, tier-classified, content-bound piece of evidence."""

    evidence_id: str
    query: str
    source_url: str
    domain: str
    tier: str
    retrieved_at: str
    claim: str
    confidence: str
    content_sha256: str
    subject_version: str = ""
    affects: str = ""
    previous_record_hash: str = GENESIS_HASH
    record_hash: str = ""

    def identity_payload(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "query": self.query,
            "source_url": self.source_url,
            "domain": self.domain,
            "tier": self.tier,
            "retrieved_at": self.retrieved_at,
            "claim": self.claim,
            "confidence": self.confidence,
            "content_sha256": self.content_sha256,
            "subject_version": self.subject_version,
            "affects": self.affects,
            "previous_record_hash": self.previous_record_hash,
        }

    def compute_hash(self) -> str:
        return sha256_payload(self.identity_payload())

    def sealed(self) -> ResearchEvidence:
        return replace(self, record_hash=self.compute_hash())

    def verify(self) -> bool:
        return bool(self.record_hash) and self.record_hash == self.compute_hash()

    @property
    def source_tier(self) -> SourceTier:
        try:
            return SourceTier(self.tier)
        except ValueError as error:
            raise EvidenceError(f"unknown source tier: {self.tier}") from error

    def matches_content(self, content: bytes) -> bool:
        """Return whether ``content`` still matches the recorded digest."""

        return hashlib.sha256(content).hexdigest() == self.content_sha256

    def to_payload(self) -> dict:
        payload = self.identity_payload()
        payload["record_hash"] = self.record_hash
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> ResearchEvidence:
        try:
            tier = str(payload["tier"])
            SourceTier(tier)
            return cls(
                evidence_id=str(payload["evidence_id"]),
                query=str(payload["query"]),
                source_url=str(payload["source_url"]),
                domain=str(payload["domain"]),
                tier=tier,
                retrieved_at=str(payload["retrieved_at"]),
                claim=str(payload["claim"]),
                confidence=str(payload["confidence"]),
                content_sha256=str(payload["content_sha256"]),
                subject_version=str(payload.get("subject_version", "")),
                affects=str(payload.get("affects", "")),
                previous_record_hash=str(payload["previous_record_hash"]),
                record_hash=str(payload.get("record_hash", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError("evidence payload is malformed") from error


def evidence_from_source(
    *,
    query: str,
    url: str,
    content: bytes,
    claim: str,
    confidence: str = "medium",
    subject_version: str = "",
    affects: str = "",
    retrieved_at: str | None = None,
    previous_record_hash: str = GENESIS_HASH,
) -> ResearchEvidence:
    """Build a sealed evidence record from retrieved bytes."""

    record = ResearchEvidence(
        evidence_id=new_identifier("evidence"),
        query=redact_secrets(query),
        source_url=redact_secrets(url),
        domain=_domain_of(url),
        tier=classify_source(url).value,
        retrieved_at=retrieved_at or utc_now_iso(),
        claim=redact_secrets(claim),
        confidence=confidence,
        content_sha256=hashlib.sha256(content).hexdigest(),
        subject_version=subject_version,
        affects=redact_secrets(affects),
        previous_record_hash=previous_record_hash,
    )
    return record.sealed()


def sufficient_for_patch(
    evidence: Sequence[ResearchEvidence],
    *,
    require_primary: bool = True,
) -> tuple:
    """Return ``(ok, reasons)`` for whether evidence may justify a patch."""

    reasons: list = []
    if not evidence:
        reasons.append("no research evidence was recorded")
        return (False, tuple(reasons))

    for item in evidence:
        if not item.verify():
            reasons.append(f"evidence {item.evidence_id} failed hash verification")

    try:
        tiers = {item.source_tier for item in evidence}
    except EvidenceError as error:
        reasons.append(str(error))
        tiers = set()

    if require_primary and SourceTier.PRIMARY not in tiers:
        present = ", ".join(sorted(t.value for t in tiers)) or "none"
        reasons.append(
            f"no PRIMARY source supports this change (only {present}); "
            "official documentation or package metadata is required"
        )

    for item in evidence:
        if contains_secret(item.claim) or contains_secret(item.source_url):
            reasons.append(f"evidence {item.evidence_id} carries credential material")

    return (not reasons, tuple(reasons))


class ResearchProvider(Protocol):
    """Dependency-injected source of evidence. Tests supply fakes."""

    def gather(self, query: str) -> Sequence[ResearchEvidence]:  # pragma: no cover
        ...


class StaticResearchProvider:
    """Deterministic provider for tests and offline Termux operation."""

    def __init__(self, evidence: Sequence[ResearchEvidence] = ()) -> None:
        self._evidence = tuple(evidence)
        self.queries: list = []

    def gather(self, query: str) -> Sequence[ResearchEvidence]:
        self.queries.append(query)
        return self._evidence


class EvidenceLedger:
    """Atomic, hash-chained, append-only store of research evidence."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict:
        if not self._path.is_file():
            return {"chain": []}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise EvidenceError("evidence ledger is unreadable") from error
        if not isinstance(raw, dict) or not isinstance(raw.get("chain"), list):
            raise EvidenceError("evidence ledger is malformed")
        return raw

    def head_hash(self) -> str:
        chain = self._read()["chain"]
        return str(chain[-1]["record_hash"]) if chain else GENESIS_HASH

    def all(self) -> tuple:
        """Return the verified chain, oldest first."""

        chain = self._read()["chain"]
        previous = GENESIS_HASH
        records = []
        for entry in chain:
            record = ResearchEvidence.from_payload(entry)
            if not record.verify():
                raise EvidenceError(f"evidence {record.evidence_id} failed hash verification")
            if record.previous_record_hash != previous:
                raise EvidenceError("evidence chain linkage is broken")
            previous = record.record_hash
            records.append(record)
        return tuple(records)

    def append(self, record: ResearchEvidence) -> ResearchEvidence:
        """Append a sealed record, rejecting unsealed or unlinked entries."""

        if not record.verify():
            raise EvidenceError("refusing to persist unsealed evidence")
        existing = self.all()
        head = existing[-1].record_hash if existing else GENESIS_HASH
        if record.previous_record_hash != head:
            raise EvidenceError("evidence does not link to the current chain head")
        chain = [item.to_payload() for item in existing]
        chain.append(record.to_payload())
        self._atomic_write({"chain": chain})
        return record

    def for_query(self, query: str) -> tuple:
        return tuple(item for item in self.all() if item.query == query)

    def _atomic_write(self, payload: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".evidence-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self._path))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise


def record_all(ledger: EvidenceLedger, evidence: Sequence[ResearchEvidence]) -> tuple:
    """Append evidence to the ledger while maintaining chain continuity."""

    stored = []
    for item in evidence:
        relinked = replace(item, previous_record_hash=ledger.head_hash(), record_hash="").sealed()
        stored.append(ledger.append(relinked))
    return tuple(stored)
