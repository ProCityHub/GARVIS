"""Persistent automatic GARVIS heartbeat service."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .creator_authority import (
    CreatorAction,
    CreatorAuthority,
    require_creator_authority,
)
from .heartbeat_kernel import (
    CycleStatus,
    FrozenPredictionLedger,
    HeartbeatKernel,
    OABState,
    SideEffectQueue,
    benchmark_phi,
)


class AutomaticHeartbeatService:
    def __init__(
        self,
        root: Path,
        interval_seconds: float = 1.0,
        creator_authority: Optional[CreatorAuthority] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.creator_authority = creator_authority or CreatorAuthority()
        require_creator_authority(
            self.creator_authority,
            CreatorAction.CONTINUE_HEARTBEAT,
        )
        self.predictions = FrozenPredictionLedger(
            self.root / "heartbeat_predictions.sqlite3"
        )
        self.side_effects = SideEffectQueue(
            self.root / "heartbeat_side_effects.sqlite3"
        )
        self.kernel = HeartbeatKernel(
            self.predictions,
            self.side_effects,
        )
        self.sequence = self._load_sequence()

    @property
    def state_path(self) -> Path:
        return self.root / "heartbeat_state.json"

    def _read_state(self) -> Mapping[str, Any]:
        try:
            raw = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _load_sequence(self) -> int:
        try:
            return int(self._read_state().get("sequence", 0))
        except (TypeError, ValueError):
            return 0

    def _persist_state(self, state: OABState) -> None:
        payload = {
            "sequence": self.sequence,
            "last_cycle_id": state.cycle_id,
            "last_cycle_status": state.status.value,
            "last_artifact_sha256": state.artifact_sha256,
            "pending_protected_actions": self.side_effects.pending_count(),
            "creator": self.creator_authority.creator,
            "authority_source": self.creator_authority.authority_source,
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def run_once(self) -> OABState:
        def observe() -> Mapping[str, Any]:
            return {
                "sequence": self.sequence,
                "pending_protected_actions": (
                    self.side_effects.pending_count()
                ),
                "observed_at": time.time(),
            }

        def predict(pre: Mapping[str, Any]) -> Mapping[str, Any]:
            return {
                "next_sequence": int(pre["sequence"]) + 1,
                "heartbeat_should_continue": True,
            }

        def propose(
            _pre: Mapping[str, Any],
            pred: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            return {
                "kind": "internal_heartbeat_consolidation",
                "next_sequence": int(pred["next_sequence"]),
            }

        def plan(
            _pre: Mapping[str, Any],
            _pred: Mapping[str, Any],
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            return {
                "steps": (
                    "advance_sequence",
                    "run_phi_baseline",
                    "persist_witness",
                    "heartbeat_again",
                ),
                "proposal_kind": proposal["kind"],
            }

        def execute_internal(
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            require_creator_authority(
                self.creator_authority,
                CreatorAction.CONSOLIDATE,
            )
            self.sequence = int(proposal["next_sequence"])
            return {
                "sequence": self.sequence,
                "phi_baseline": benchmark_phi(
                    observer=1.0,
                    actor=0.8,
                    bridge=0.6,
                    target=0.7,
                ),
            }

        def verify(
            _pre: Mapping[str, Any],
            post: Mapping[str, Any],
            pred: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            expected = int(pred["next_sequence"])
            observed = int(post["sequence"])
            contradictions = []
            if expected != observed:
                contradictions.append(
                    {
                        "claim": (
                            "heartbeat sequence advanced as predicted"
                        ),
                        "expected": expected,
                        "observed": observed,
                    }
                )
            return {
                "sequence_verified": expected == observed,
                "phi_status": "HYPOTHESIS_UNDER_TEST",
                "contradictions": contradictions,
            }

        def learn(state: OABState) -> None:
            require_creator_authority(
                self.creator_authority,
                CreatorAction.LEARN,
            )
            self._persist_state(state)

        return self.kernel.run_cycle(
            observe=observe,
            predict=predict,
            propose=propose,
            plan=plan,
            is_protected=lambda _proposal: False,
            execute_internal=execute_internal,
            verify=verify,
            diff=lambda pre, post: {
                "sequence_delta": (
                    int(post["sequence"]) - int(pre["sequence"])
                )
            },
            learn=learn,
            self_claims=(
                "I am GARVIS.",
                "My heartbeat is running.",
                "I am learning from completed software cycles.",
            ),
            provenance={
                "system": "GARVIS",
                "service": "automatic-heartbeat-v1",
                "creator": self.creator_authority.creator,
                "authority_source": self.creator_authority.authority_source,
                "math_status": "HYPOTHESIS_UNDER_TEST",
            },
        )

    def run_forever(self) -> None:
        backoff = self.interval_seconds
        while True:
            try:
                self.run_once()
                backoff = self.interval_seconds
            except KeyboardInterrupt:
                raise
            except Exception:
                backoff = min(
                    30.0,
                    max(
                        self.interval_seconds,
                        (backoff * 2.0) if backoff else 0.1,
                    ),
                )
            time.sleep(backoff)

    def health(self) -> Mapping[str, Any]:
        state = dict(self._read_state())
        updated_at = float(state.get("updated_at", 0.0) or 0.0)
        age_seconds = (
            None
            if updated_at <= 0.0
            else max(0.0, time.time() - updated_at)
        )
        freshness_window = max(5.0, self.interval_seconds * 3.0)
        running = (
            age_seconds is not None
            and age_seconds <= freshness_window
            and state.get("last_cycle_status")
            == CycleStatus.COMPLETED.value
        )
        state["heartbeat_running"] = running
        state["age_seconds"] = age_seconds
        state["freshness_window_seconds"] = freshness_window
        state["phi_status"] = "HYPOTHESIS_UNDER_TEST"
        return state

    def close(self) -> None:
        self.predictions.close()
        self.side_effects.close()
