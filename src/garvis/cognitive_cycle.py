"""DIRECTIVE-010: Cognitive cycle engine for GARVIS.

Runs one full cognitive cycle and emits a snapshot that passes
``validate_hypercube_snapshot``. Prior snapshots in the snapshot
directory are recalled and fed into the next cycle's observation,
giving the engine continuity across cycles.

Governance: ``power_request`` defaults to not requested. Any cycle
that requests power escalation HALTS the engine until an external
approval callable confirms. No auto-approval path exists.

Adrien D. Thomas retains merge and external-action authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .hypercube_snapshot import validate_hypercube_snapshot

CYCLE_VERSION = "1.0"


class CycleHaltedError(RuntimeError):
    """Raised when the engine is halted pending external approval."""


class CycleEngine:
    """Executes cognitive cycles and records them as snapshots on disk."""

    def __init__(
        self,
        snapshot_dir: Path,
        operator_context: str = "adrien",
        recall_depth: int = 5,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.operator_context = operator_context
        self.recall_depth = recall_depth
        self.halted = False
        self._pending: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Memory: the diary of prior cycles.
    # ------------------------------------------------------------------
    def _snapshot_paths(self) -> List[Path]:
        return sorted(self.snapshot_dir.glob("cycle-*.json"))

    def next_cycle_id(self) -> str:
        """Deterministic, monotonically increasing cycle id."""
        return f"cycle-{len(self._snapshot_paths()) + 1:06d}"

    def recall(self) -> List[Dict[str, Any]]:
        """Load the most recent snapshots (oldest first within the window)."""
        memories: List[Dict[str, Any]] = []
        for path in self._snapshot_paths()[-self.recall_depth :]:
            memories.append(json.loads(path.read_text()))
        return memories

    # ------------------------------------------------------------------
    # The cycle itself.
    # ------------------------------------------------------------------
    def run_cycle(
        self,
        input_state: Mapping[str, Any],
        candidate_thoughts: List[str],
        requests_power: bool = False,
        power_reason: str = "",
    ) -> Dict[str, Any]:
        """Run one cycle: observe, compare, select, record.

        Returns the validated snapshot. If ``requests_power`` is True the
        snapshot is recorded with status ``halted-pending-approval`` and the
        engine halts; ``CycleHaltedError`` is raised on the *next* call.
        """
        if self.halted:
            raise CycleHaltedError(
                "Engine halted pending approval of "
                f"{(self._pending or {}).get('cycle_id', 'unknown cycle')}"
            )
        if not candidate_thoughts:
            raise ValueError("A cycle requires at least one candidate thought.")

        memories = self.recall()
        prior_selections = [m["selection"]["chosen"] for m in memories]

        observation = {
            "input_keys": sorted(input_state.keys()),
            "remembered_cycles": [m["cycle_id"] for m in memories],
            "prior_selections": prior_selections,
        }

        # Compare: prefer continuity — a thought consistent with the most
        # recent selection scores higher; otherwise first candidate wins.
        scores: Dict[str, int] = {}
        last = prior_selections[-1] if prior_selections else None
        for idx, thought in enumerate(candidate_thoughts):
            scores[thought] = (2 if last and last in thought else 1) * 1000 - idx
        chosen = max(candidate_thoughts, key=lambda t: scores[t])

        uncertainty = round(1.0 / (1 + len(memories)), 4)

        snapshot: Dict[str, Any] = {
            "cycle_id": self.next_cycle_id(),
            "cycle_version": CYCLE_VERSION,
            "status": "halted-pending-approval" if requests_power else "complete",
            "stage": "selection",
            "operator_context": {"operator": self.operator_context},
            "input_state": dict(input_state),
            "observation_summary": observation,
            "candidate_thoughts": list(candidate_thoughts),
            "comparison": {"scores": scores},
            "selection": {"chosen": chosen},
            "uncertainty": {"value": uncertainty},
            "evolution_contract": {
                "constitution": "autonomous in viewpoint, not in action",
                "amendable_by": self.operator_context,
            },
            "next_smallest_step": {
                "step": "await-approval" if requests_power else f"act-on:{chosen}"
            },
            "output_boundary": {"external_actions": False},
            "power_request": {
                "power_requested": bool(requests_power),
                "requested_permissions": [power_reason] if requests_power else [],
                "why_power_should_be_refused": (
                    "Constitution: autonomous in viewpoint, not in action. "
                    "Escalation is refused by default."
                ),
                "approval_required": True,
                "ledger_required": True,
            },
        }

        validate_hypercube_snapshot(snapshot)
        path = self.snapshot_dir / f"{snapshot['cycle_id']}.json"
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))

        if requests_power:
            self.halted = True
            self._pending = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # Governance: resumption requires external approval. Never automatic.
    # ------------------------------------------------------------------
    def resume(self, approve: Callable[[Dict[str, Any]], bool]) -> None:
        """Resume a halted engine.

        ``approve`` is an external check (e.g. a one-shot approval token
        verifier). The engine never approves itself: if the callable does
        not return True, the engine stays halted.
        """
        if not self.halted:
            return
        if self._pending is not None and approve(self._pending) is True:
            self.halted = False
            self._pending = None
        else:
            raise CycleHaltedError("Approval denied; engine remains halted.")
