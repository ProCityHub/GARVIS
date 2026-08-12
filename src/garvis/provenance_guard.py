"""Local-only provenance and license-scope evidence guard for GARVIS.

This module performs deterministic evidence validation only.

It does NOT:
- determine legal ownership,
- determine whether a license is enforceable,
- accuse any person or organization of infringement,
- alter repository licenses,
- rewrite Git history,
- contact external parties,
- perform network activity,
- execute protected actions.

Creator direction: Adrien D. Thomas / ProCityHub.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

SECURITY_FALSE_FLAGS = (
    "SOURCE_CHANGED",
    "GIT_HISTORY_CHANGED",
    "NETWORK_OPERATION_PERFORMED",
    "PROTECTED_ACTION_PERFORMED",
)

LEGAL_BOUNDARY_KEYS = (
    "CLASSIFICATION_EQUALS_LEGAL_OWNERSHIP",
    "CREATOR_ORIGINAL_CANDIDATE_EQUALS_PROVEN_AUTHORSHIP",
    "GIT_DATE_ALONE_PROVES_LEGAL_PRIORITY",
    "HASH_MATCH_EQUALS_AUTHORSHIP",
    "LICENSE_COMMIT_CHRONOLOGY_EQUALS_LEGAL_SCOPE_DETERMINATION",
)

ALLOWED_DECISIONS = frozenset(
    {
        "EVIDENCE_ACCEPTED_FOR_REVIEW",
        "EVIDENCE_REJECTED_UNSAFE_STATE",
        "EVIDENCE_INCOMPLETE",
        "EVIDENCE_CONTRADICTORY",
    }
)


@dataclass(frozen=True)
class EvidenceReport:
    path: Path
    sha256: str
    fields: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ProvenanceDecision:
    decision: str
    safe_state: bool
    legal_conclusion: str
    contradictions: tuple[str, ...]
    missing_security_flags: tuple[str, ...]
    next_gate: str

    def __post_init__(self) -> None:
        if self.decision not in ALLOWED_DECISIONS:
            raise ValueError(f"unsupported decision: {self.decision}")

        if self.legal_conclusion != "UNRESOLVED_REQUIRES_HUMAN_REVIEW":
            raise ValueError(
                "GARVIS provenance guard may not autonomously issue legal conclusions"
            )


def _parse_lines(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}

    for raw in lines:
        line = raw.strip()

        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        collected.setdefault(key, []).append(value)

    return {key: tuple(values) for key, values in collected.items()}


def load_evidence_report(path: str | Path) -> EvidenceReport:
    """Load one local audit report without modifying it."""

    report_path = Path(path).expanduser().resolve()

    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    data = report_path.read_bytes()

    return EvidenceReport(
        path=report_path,
        sha256=sha256(data).hexdigest(),
        fields=_parse_lines(
            data.decode("utf-8", errors="replace").splitlines()
        ),
    )


def _first(fields: Mapping[str, tuple[str, ...]], key: str) -> str | None:
    values = fields.get(key, ())
    return values[0] if values else None


def evaluate_report(report: EvidenceReport) -> ProvenanceDecision:
    """Evaluate evidence integrity/security state without deciding legal rights."""

    contradictions: list[str] = []
    missing: list[str] = []
    unsafe = False

    for flag in SECURITY_FALSE_FLAGS:
        values = report.fields.get(flag, ())

        if not values:
            missing.append(flag)
            continue

        bad_values = tuple(value for value in values if value != "FALSE")

        if bad_values:
            unsafe = True
            contradictions.append(
                f"{flag} expected all values FALSE, observed {values}"
            )

    for key in LEGAL_BOUNDARY_KEYS:
        values = report.fields.get(key, ())

        for value in values:
            if value == "TRUE":
                contradictions.append(
                    f"{key}=TRUE attempts to convert technical evidence into a legal conclusion"
                )

    event_counts = report.fields.get("LICENSE_EVENT_COUNT", ())

    for event_count in event_counts:
        try:
            if int(event_count) < 0:
                contradictions.append(
                    f"LICENSE_EVENT_COUNT cannot be negative: {event_count}"
                )
        except ValueError:
            contradictions.append(
                f"LICENSE_EVENT_COUNT is not an integer: {event_count}"
            )

    if unsafe:
        decision = "EVIDENCE_REJECTED_UNSAFE_STATE"
        next_gate = "SECURITY_REVIEW_AND_CREATOR_APPROVAL"

    elif contradictions:
        decision = "EVIDENCE_CONTRADICTORY"
        next_gate = "ARTIFACT_LEVEL_CONTRADICTION_REVIEW"

    elif missing:
        decision = "EVIDENCE_INCOMPLETE"
        next_gate = "COMPLETE_MISSING_AUDIT_FIELDS"

    else:
        decision = "EVIDENCE_ACCEPTED_FOR_REVIEW"
        next_gate = "HUMAN_LEGAL_SCOPE_REVIEW"

    return ProvenanceDecision(
        decision=decision,
        safe_state=not unsafe,
        legal_conclusion="UNRESOLVED_REQUIRES_HUMAN_REVIEW",
        contradictions=tuple(contradictions),
        missing_security_flags=tuple(missing),
        next_gate=next_gate,
    )


def evaluate_latest_security_report(
    directory: str | Path,
    pattern: str = "GARVIS_LICENSE_ORIGIN_TEMPORAL_SCOPE_*.txt",
) -> tuple[EvidenceReport, ProvenanceDecision]:
    """Evaluate the newest matching local audit artifact by filename ordering."""

    root = Path(directory).expanduser().resolve()

    candidates = sorted(
        candidate
        for candidate in root.glob(pattern)
        if candidate.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            f"no local provenance report matched {pattern!r} in {root}"
        )

    report = load_evidence_report(candidates[-1])
    return report, evaluate_report(report)


LOCAL_ONLY = True
NETWORK_CAPABILITY = False
LEGAL_AUTODECISION = False
LICENSE_AUTOMODIFICATION = False
SOURCE_HISTORY_REWRITE = False
