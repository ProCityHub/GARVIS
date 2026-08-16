"""Command-line access to GARVIS Hypercube Heartbeat mathematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from garvis.hypercube_heartbeat_math import (
    PERSPECTIVES,
    SemanticOAB,
    boundary_overlap,
    heartbeat_angle,
    invariant_report,
    mirror_phase_index,
    observer_gain,
    oab_wrap_phase_index,
    phase_coordinate,
    reciprocal_phase_index,
    semantic_barycenter,
    semantic_coupling_normalized,
    semantic_coupling_raw,
    semantic_dirichlet_energy,
    semantic_elasticities,
    semantic_entropy_bits,
)
from garvis.hypercube_quantum_metrics import compare_distributions


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m garvis.hypercube_math_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("structure", help="Print exact Hypercube/Heartbeat invariants.")

    phase = sub.add_parser("phase", help="Inspect one Heartbeat phase index.")
    phase.add_argument("index", type=int)

    oab = sub.add_parser("oab", help="Evaluate the semantic O/A/B framework descriptor.")
    oab.add_argument("observer", type=float)
    oab.add_argument("actor", type=float)
    oab.add_argument("background", type=float)

    field = sub.add_parser("field", help="Analyze an eight-perspective semantic field JSON.")
    field.add_argument("json_path", type=Path)

    quantum = sub.add_parser("quantum", help="Recompute metrics from a V18 evidence JSON.")
    quantum.add_argument("json_path", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "structure":
        _print(invariant_report())
        return 0

    if args.command == "phase":
        k = args.index % 9
        _print(
            {
                "index": k,
                "coordinate": phase_coordinate(k),
                "angle_radians": heartbeat_angle(k),
                "mirror_index": mirror_phase_index(k),
                "wrap_index": oab_wrap_phase_index(k),
                "reciprocal_index": reciprocal_phase_index(k),
                "boundary_overlap": boundary_overlap(phase_coordinate(k)),
                "observer_gain": observer_gain(phase_coordinate(k)),
            }
        )
        return 0

    if args.command == "oab":
        state = SemanticOAB(args.observer, args.actor, args.background)
        _print(
            {
                "inputs": {
                    "observer": state.observer,
                    "actor": state.actor,
                    "background": state.background,
                },
                "raw_descriptor": semantic_coupling_raw(state),
                "degree_normalized_descriptor": semantic_coupling_normalized(state),
                "elasticities": semantic_elasticities(),
                "decision_score": False,
            }
        )
        return 0

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))

    if args.command == "field":
        if not isinstance(payload, dict):
            raise SystemExit("field JSON must be an object mapping perspective codes to weights")
        unknown = set(payload).difference(PERSPECTIVES)
        if unknown:
            raise SystemExit("unknown perspective codes: {}".format(", ".join(sorted(unknown))))
        _print(
            {
                "barycenter": semantic_barycenter(payload),
                "entropy_bits": semantic_entropy_bits(payload),
                "dirichlet_energy": semantic_dirichlet_energy(payload),
            }
        )
        return 0

    if args.command == "quantum":
        ideal = payload.get("ideal_probabilities")
        counts = payload.get("observed_counts")
        if not isinstance(ideal, dict) or not isinstance(counts, dict):
            raise SystemExit("evidence JSON must contain ideal_probabilities and observed_counts")
        shots = sum(int(value) for value in counts.values())
        observed = {key: int(value) / float(shots) for key, value in counts.items()}
        _print(compare_distributions(ideal, observed))
        return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
