"""Local JSON adapter for the deterministic GARVIS lattice cycle.

Authored under the direction of Adrien D. Thomas, operating as ProCityHub.

This adapter reads an explicitly supplied evidence file and returns a bounded,
machine-readable cognitive-cycle result. It does not contact an LLM, browse a
network, execute an outside-world action, or convert conversation into evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evidence_envelope import EvidenceEnvelope, build_evidence_envelope
from .lattice_cognition import run_lattice_cognitive_cycle


_ENVELOPE_FIELDS = {
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
}


def _mapping_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    if field_name not in payload:
        if required:
            raise ValueError(
                f"evidence JSON requires {field_name!r}"
            )
        return {}

    value = payload[field_name]

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name!r} must contain a JSON object"
        )

    return dict(value)


def load_evidence_envelope(path: Path) -> EvidenceEnvelope:
    """Load and validate one explicit JSON evidence envelope."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"unable to read evidence file {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"evidence file is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError(
            "evidence JSON must contain one top-level object"
        )

    unknown_fields = sorted(
        set(payload) - _ENVELOPE_FIELDS
    )

    if unknown_fields:
        raise ValueError(
            "unsupported evidence fields: "
            + ", ".join(unknown_fields)
        )

    hypercube_data = payload.get("hypercube_cycle_data")

    if (
        hypercube_data is not None
        and not isinstance(hypercube_data, Mapping)
    ):
        raise ValueError(
            "'hypercube_cycle_data' must be an object or null"
        )

    return build_evidence_envelope(
        immutable_source_evidence=_mapping_field(
            payload,
            "immutable_source_evidence",
            required=True,
        ),
        hypercube_cycle_data=(
            None
            if hypercube_data is None
            else dict(hypercube_data)
        ),
        deterministic_agi_measurements=_mapping_field(
            payload,
            "deterministic_agi_measurements",
        ),
        scientific_claims=_mapping_field(
            payload,
            "scientific_claims",
        ),
        source_provenance_hashes=_mapping_field(
            payload,
            "source_provenance_hashes",
        ),
        approval_state=_mapping_field(
            payload,
            "approval_state",
        ),
        garvis_evidence_summary=_mapping_field(
            payload,
            "garvis_evidence_summary",
        ),
        engineering_inference=_mapping_field(
            payload,
            "engineering_inference",
        ),
        symbolic_interpretation=_mapping_field(
            payload,
            "symbolic_interpretation",
        ),
        unsupported_speculation=_mapping_field(
            payload,
            "unsupported_speculation",
        ),
        unknowns=_mapping_field(
            payload,
            "unknowns",
        ),
    )


def run_lattice_cycle_file(
    *,
    path: Path,
    cycle: int,
    external_action: bool,
) -> dict[str, Any]:
    """Run one local deterministic cycle and return JSON-ready output."""

    envelope = load_evidence_envelope(path)

    result = run_lattice_cognitive_cycle(
        envelope=envelope,
        cycle=cycle,
        external_action=external_action,
    )

    equilibrium = result.recall_equilibrium.equilibrium

    return {
        "mode": "local_lattice_cycle",
        "cycle": result.cycle,
        "decision": result.decision,
        "proposal_eligible": result.proposal_eligible,
        "human_approval_required": (
            result.human_approval_required
        ),
        "external_action_requested": bool(external_action),
        "external_action_allowed": (
            result.external_action_allowed
        ),
        "cycle_sha256": result.cycle_sha256,
        "envelope_sha256": result.envelope_sha256,
        "evidence": {
            "sufficiency": (
                result.assessment.evidence_sufficiency
            ),
            "signal_count": len(result.assessment.signals),
            "retained_signal_count": len(
                result.consolidated_memory.retained_signal_ids
            ),
            "excluded_signal_count": len(
                result.consolidated_memory.excluded_signal_ids
            ),
        },
        "pulse": {
            "phase": result.pulse.phase.value,
            "raw_union": result.pulse.raw_union,
            "normalized_center": (
                result.pulse.normalized_center
            ),
            "sha256": result.pulse.compute_sha256(),
        },
        "recall": {
            "recalled": result.recall.recalled,
            "converged": result.recall.converged,
            "similarity": result.recall.similarity,
            "cycles_run": result.recall.cycles_run,
            "final_delta": result.recall.final_delta,
            "sha256": (
                result.recall_equilibrium.recall_sha256
            ),
        },
        "equilibrium": {
            "raw_union": equilibrium.raw_union,
            "normalized_center": (
                equilibrium.normalized_center
            ),
            "psychological_coherence": (
                equilibrium.psychological_coherence
            ),
            "integrated_equilibrium": (
                equilibrium.integrated_equilibrium
            ),
            "corner_bits": list(equilibrium.corner_bits),
            "corner_index": equilibrium.corner_index,
            "equilibrium_reached": (
                equilibrium.equilibrium_reached
            ),
            "limiting_dimension": (
                equilibrium.limiting_dimension
            ),
        },
        "completed_stages": [
            stage.value
            for stage in result.completed_stages
        ],
        "claim_boundary": result.claim_boundary,
    }
