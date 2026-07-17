"""Read-only evidence envelope for GARVIS system data.

This module keeps validated source evidence separate from GARVIS-generated
inference, symbolic interpretation, speculation, and unknowns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from garvis.hypercube_snapshot import (
    HypercubeSnapshotError,
    validate_hypercube_snapshot,
)


# Stable validation seam for tests and future read-only integrations.
validate_snapshot = validate_hypercube_snapshot



class FrozenList(tuple):
    """Immutable sequence that retains JSON-list equality semantics."""

    def __new__(cls, values: Any = ()) -> "FrozenList":
        return super().__new__(cls, values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return False

    __hash__ = tuple.__hash__


def _freeze(value: Any) -> Any:
    """Recursively convert mutable containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return FrozenList(_freeze(item) for item in value)

    if isinstance(value, tuple):
        return FrozenList(_freeze(item) for item in value)

    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)

    return value


def _plain(value: Any) -> Any:
    """Convert immutable containers into JSON-serializable plain values."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_plain(item) for item in value]

    if isinstance(value, frozenset):
        return sorted(_plain(item) for item in value)

    return value


@dataclass(frozen=True)
class EvidenceEnvelope:
    """Immutable container separating evidence from generated interpretation."""

    immutable_source_evidence: Mapping[str, Any] = field(default_factory=dict)
    hypercube_cycle_data: Mapping[str, Any] | None = None
    deterministic_agi_measurements: Mapping[str, Any] = field(
        default_factory=dict
    )
    scientific_claims: Mapping[str, Any] = field(default_factory=dict)
    source_provenance_hashes: Mapping[str, Any] = field(default_factory=dict)
    approval_state: Mapping[str, Any] = field(default_factory=dict)
    garvis_evidence_summary: Mapping[str, Any] = field(default_factory=dict)
    engineering_inference: Mapping[str, Any] = field(default_factory=dict)
    symbolic_interpretation: Mapping[str, Any] = field(default_factory=dict)
    unsupported_speculation: Mapping[str, Any] = field(default_factory=dict)
    unknowns: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hypercube_cycle_data is not None:
            try:
                validate_snapshot(self.hypercube_cycle_data)
            except HypercubeSnapshotError as exc:
                raise ValueError(
                    f"Invalid hypercube_cycle_data provided: {exc}"
                ) from exc

        immutable_fields = (
            "immutable_source_evidence",
            "hypercube_cycle_data",
            "deterministic_agi_measurements",
            "scientific_claims",
            "source_provenance_hashes",
            "approval_state",
            "garvis_evidence_summary",
            "engineering_inference",
            "symbolic_interpretation",
            "unsupported_speculation",
            "unknowns",
        )

        for field_name in immutable_fields:
            value = getattr(self, field_name)

            if value is not None:
                object.__setattr__(self, field_name, _freeze(value))

    def compute_sha256_hash(self, data: Any) -> str:
        """Compute a deterministic SHA-256 hash for JSON-compatible data."""

        encoded = json.dumps(
            _plain(data),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def with_source_provenance(
        self,
        source_key: str,
        source_data: Any,
    ) -> EvidenceEnvelope:
        """Return a new envelope containing a source-provenance hash."""

        updated_hashes = dict(self.source_provenance_hashes)
        updated_hashes[source_key] = self.compute_sha256_hash(source_data)

        return replace(
            self,
            source_provenance_hashes=updated_hashes,
        )


def build_evidence_envelope(
    *,
    immutable_source_evidence: Mapping[str, Any],
    hypercube_cycle_data: Mapping[str, Any] | None = None,
    deterministic_agi_measurements: Mapping[str, Any] | None = None,
    scientific_claims: Mapping[str, Any] | None = None,
    source_provenance_hashes: Mapping[str, Any] | None = None,
    approval_state: Mapping[str, Any] | None = None,
    garvis_evidence_summary: Mapping[str, Any] | None = None,
    engineering_inference: Mapping[str, Any] | None = None,
    symbolic_interpretation: Mapping[str, Any] | None = None,
    unsupported_speculation: Mapping[str, Any] | None = None,
    unknowns: Mapping[str, Any] | None = None,
) -> EvidenceEnvelope:
    """Build an immutable evidence envelope from read-only input records."""

    return EvidenceEnvelope(
        immutable_source_evidence=dict(immutable_source_evidence),
        hypercube_cycle_data=(
            None
            if hypercube_cycle_data is None
            else dict(hypercube_cycle_data)
        ),
        deterministic_agi_measurements=dict(
            deterministic_agi_measurements or {}
        ),
        scientific_claims=dict(scientific_claims or {}),
        source_provenance_hashes=dict(source_provenance_hashes or {}),
        approval_state=dict(approval_state or {}),
        garvis_evidence_summary=dict(garvis_evidence_summary or {}),
        engineering_inference=dict(engineering_inference or {}),
        symbolic_interpretation=dict(symbolic_interpretation or {}),
        unsupported_speculation=dict(unsupported_speculation or {}),
        unknowns=dict(unknowns or {}),
    )
