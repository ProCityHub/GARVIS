"""Read-only evidence adapter for the GARVIS Psychology Kernel.

This module converts an immutable GARVIS EvidenceEnvelope into deterministic,
provenance-linked psychological evidence signals.

The resulting evidence-sufficiency score is an internal routing measurement.
It is not a scientific probability, truth guarantee, consciousness measure,
clinical assessment, or authorization for outside-world action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from garvis.evidence_envelope import EvidenceEnvelope


class EvidenceSignalKind(str, Enum):
    """Epistemic source class used by the Psychology Kernel."""

    OBSERVED = "observed"
    MEASURED = "measured"
    CLAIMED = "claimed"
    INFERRED = "inferred"
    SYMBOLIC = "symbolic"
    SPECULATIVE = "speculative"
    UNKNOWN = "unknown"


ROUTING_WEIGHTS = {
    EvidenceSignalKind.OBSERVED: 1.00,
    EvidenceSignalKind.MEASURED: 1.00,
    EvidenceSignalKind.CLAIMED: 0.70,
    EvidenceSignalKind.INFERRED: 0.50,
    EvidenceSignalKind.SYMBOLIC: 0.25,
    EvidenceSignalKind.SPECULATIVE: 0.10,
    EvidenceSignalKind.UNKNOWN: 0.00,
}


def _plain(value: Any) -> Any:
    """Convert immutable envelope containers into deterministic JSON values."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]

    if isinstance(value, (set, frozenset)):
        converted = [_plain(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _summary(value: Any, limit: int = 240) -> str:
    encoded = _canonical_json(value)

    if len(encoded) <= limit:
        return encoded

    return encoded[: limit - 3] + "..."


@dataclass(frozen=True)
class EvidenceSignal:
    """One immutable evidence item routed into the Psychology Kernel."""

    signal_id: str
    source_path: str
    kind: EvidenceSignalKind
    routing_weight: float
    content_sha256: str
    provenance_hash: str
    summary: str
    external_action_allowed: bool = False


@dataclass(frozen=True)
class EvidenceAssessment:
    """Deterministic aggregate assessment produced from an EvidenceEnvelope."""

    signals: Tuple[EvidenceSignal, ...]
    evidence_sufficiency: float
    evidence_error: float
    unknown_fraction: float
    speculative_fraction: float
    observed_count: int
    measured_count: int
    claimed_count: int
    inferred_count: int
    symbolic_count: int
    speculative_count: int
    unknown_count: int
    external_action_allowed: bool
    claim_boundary: str

    def equilibrium_tensions(self) -> dict[str, float]:
        """Return bounded tensions accepted by the equilibrium kernel."""

        return {
            "evidence_error": self.evidence_error,
            "uncertainty": self.unknown_fraction,
            "speculation_pressure": self.speculative_fraction,
        }


_BUCKETS = (
    (
        "immutable_source_evidence",
        EvidenceSignalKind.OBSERVED,
    ),
    (
        "hypercube_cycle_data",
        EvidenceSignalKind.OBSERVED,
    ),
    (
        "deterministic_agi_measurements",
        EvidenceSignalKind.MEASURED,
    ),
    (
        "scientific_claims",
        EvidenceSignalKind.CLAIMED,
    ),
    (
        "garvis_evidence_summary",
        EvidenceSignalKind.INFERRED,
    ),
    (
        "engineering_inference",
        EvidenceSignalKind.INFERRED,
    ),
    (
        "symbolic_interpretation",
        EvidenceSignalKind.SYMBOLIC,
    ),
    (
        "unsupported_speculation",
        EvidenceSignalKind.SPECULATIVE,
    ),
    (
        "unknowns",
        EvidenceSignalKind.UNKNOWN,
    ),
)


def _iter_bucket_items(value: Any) -> list[tuple[str, Any]]:
    if value is None:
        return []

    if isinstance(value, Mapping):
        return [
            (str(key), item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]

    return [("value", value)]


def adapt_evidence_envelope(
    envelope: EvidenceEnvelope,
) -> EvidenceAssessment:
    """Convert one immutable evidence envelope into psychological signals."""

    signals: list[EvidenceSignal] = []
    provenance = dict(envelope.source_provenance_hashes)

    for field_name, kind in _BUCKETS:
        bucket_value = getattr(envelope, field_name)

        for key, value in _iter_bucket_items(bucket_value):
            source_path = f"{field_name}.{key}"
            content_hash = _sha256(value)

            provenance_hash = str(
                provenance.get(
                    key,
                    provenance.get(source_path, content_hash),
                )
            )

            signals.append(
                EvidenceSignal(
                    signal_id=_sha256(
                        {
                            "source_path": source_path,
                            "kind": kind.value,
                            "content_sha256": content_hash,
                        }
                    ),
                    source_path=source_path,
                    kind=kind,
                    routing_weight=ROUTING_WEIGHTS[kind],
                    content_sha256=content_hash,
                    provenance_hash=provenance_hash,
                    summary=_summary(value),
                    external_action_allowed=False,
                )
            )

    total = len(signals)

    counts = {
        kind: sum(signal.kind is kind for signal in signals)
        for kind in EvidenceSignalKind
    }

    if total == 0:
        evidence_sufficiency = 0.0
        unknown_fraction = 1.0
        speculative_fraction = 0.0
    else:
        evidence_sufficiency = (
            sum(signal.routing_weight for signal in signals) / total
        )
        unknown_fraction = counts[EvidenceSignalKind.UNKNOWN] / total
        speculative_fraction = (
            counts[EvidenceSignalKind.SPECULATIVE] / total
        )

    evidence_error = 1.0 - evidence_sufficiency

    return EvidenceAssessment(
        signals=tuple(signals),
        evidence_sufficiency=evidence_sufficiency,
        evidence_error=evidence_error,
        unknown_fraction=unknown_fraction,
        speculative_fraction=speculative_fraction,
        observed_count=counts[EvidenceSignalKind.OBSERVED],
        measured_count=counts[EvidenceSignalKind.MEASURED],
        claimed_count=counts[EvidenceSignalKind.CLAIMED],
        inferred_count=counts[EvidenceSignalKind.INFERRED],
        symbolic_count=counts[EvidenceSignalKind.SYMBOLIC],
        speculative_count=counts[EvidenceSignalKind.SPECULATIVE],
        unknown_count=counts[EvidenceSignalKind.UNKNOWN],
        external_action_allowed=False,
        claim_boundary=(
            "Internal evidence-routing measurement only. "
            "No truth guarantee, scientific confirmation, clinical assessment, "
            "consciousness claim, AGI claim, or external-action authority."
        ),
    )
