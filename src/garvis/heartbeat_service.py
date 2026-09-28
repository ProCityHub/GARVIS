"""Persistent automatic GARVIS heartbeat service."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .creator_authority import CreatorAuthority
from .brain_binding import HeartbeatBrainBinding
from .self_authority import GarvisSelfAuthority, InternalAction, require_self_authority
from .heartbeat_kernel import (
    CycleStatus,
    HeartbeatKernel,
    OABState,
    PredictionWitnessLedger,
    SideEffectQueue,
    benchmark_phi,
)
from .heartbeat_self_dialogue import (
    build_internal_dialogue,
    observe_system,
)
from .prime_oab_reciprocal import (
    HEARTBEAT_PHASE_COUNT,
    PHASE_NAMES,
    oab_wrap_phase_index,
)


class AutomaticHeartbeatService:
    def __init__(
        self,
        root: Path,
        interval_seconds: float = 1.0,
        creator_authority: Optional[CreatorAuthority] = None,
        self_authority: Optional[GarvisSelfAuthority] = None,
        repository_root: Optional[Path] = None,
        speak: bool = False,
    ) -> None:
        """Open local ledgers, restore the checkpoint, and reconcile committed results."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.creator_authority = creator_authority or CreatorAuthority()
        self.self_authority = self_authority or GarvisSelfAuthority()
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root is not None
            else Path.cwd().resolve()
        )
        self.speak = bool(speak)
        self._last_spoken_digest = ""

        require_self_authority(
                self.self_authority,
                InternalAction.HEARTBEAT,
            )

        self.brain = HeartbeatBrainBinding(self._read_state().get("brain_checkpoint"))
        self.predictions = PredictionWitnessLedger(
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
        self.rebound_count = self._load_rebound_count()
        saved = self._read_state()
        # Older checkpoints identify their last incorporated cycle by UUID.
        anchor = self.predictions.db.execute(
            "SELECT MAX(r.rowid) FROM results r JOIN frozen_predictions p "
            "ON p.prediction_id = r.prediction_id WHERE p.cycle_id = ?",
            (saved.get("last_cycle_id"),),
        ).fetchone()[0]
        self._brain_result_rowid = int(saved.get("brain_result_rowid", anchor or 0))
        try:
            self._recover_brain_results()
        except BaseException:
            self.close()
            raise

    @property
    def state_path(self) -> Path:
        """Return the atomic heartbeat checkpoint path."""
        return self.root / "heartbeat_state.json"

    def _read_state(self) -> Mapping[str, Any]:
        """Read the checkpoint, treating missing or unreadable JSON as empty state."""
        try:
            raw = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _load_sequence(self) -> int:
        """Restore the saved heartbeat sequence or start at zero."""
        try:
            return int(self._read_state().get("sequence", 0))
        except (TypeError, ValueError):
            return 0

    def _load_rebound_count(self) -> int:
        """Restore the recorded retry count or start at zero."""
        try:
            return int(self._read_state().get("rebound_count", 0))
        except (TypeError, ValueError):
            return 0

    def _persist_state(self, state: OABState) -> None:
        """Save cycle health, learning counters, and the consumed-result cursor together."""
        raw_pre = state.raw_pre if isinstance(state.raw_pre, Mapping) else {}
        raw_post = state.raw_post if isinstance(state.raw_post, Mapping) else {}

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
            "heartbeat_version": "v2-smart",
            "current_phase": "RECEIVE",
            "phase_flow": list(PHASE_NAMES) + ["RECEIVE"],
            "alpha_omega_closure": "CONSOLIDATE->RECEIVE",
            "witness_capture": "automatic_non_blocking",
            "rebound_count": self.rebound_count,
            "system_observation": raw_pre.get("system", {}),
            "internal_dialogue": raw_post.get(
                "internal_dialogue",
                state.provenance.get("internal_dialogue", {}),
            ),
            "phi_status": "HYPOTHESIS_UNDER_TEST",
            "brain": raw_post.get("brain", state.prediction.get("brain", {})),
            "brain_checkpoint": self.brain.checkpoint(),
            "brain_result_rowid": self._brain_result_rowid,
            "contradictions": list(state.contradictions),
        }

        self._write_state(payload)

    def _write_state(self, payload: Mapping[str, Any]) -> None:
        """Atomically replace the checkpoint and its incorporated-result cursor."""
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _recover_brain_results(self, persist: bool = True) -> None:
        """Replay committed results after the checkpoint without repeating actions.

        Result row IDs and the brain counters are saved together. Interrupted
        replay is safe to repeat from the prior checkpoint. Frozen predictions
        without a committed result cannot train the brain. One service owns a
        state directory; concurrent writers are not supported.
        """
        saved = dict(self._read_state())
        changed = False
        rows = self.predictions.db.execute(
            "SELECT r.rowid, p.cycle_id, p.payload_json, r.payload_json, r.recorded_at "
            "FROM results r JOIN frozen_predictions p "
            "ON p.prediction_id = r.prediction_id WHERE r.rowid > ? ORDER BY r.rowid",
            (self._brain_result_rowid,),
        )
        for rowid, cycle_id, prediction_json, result_json, recorded_at in rows:
            prediction = json.loads(prediction_json)
            result = json.loads(result_json)
            brain = prediction.get("brain")
            if not isinstance(brain, dict):
                self._brain_result_rowid = rowid
                continue
            verification = result.get("verification", {})
            eligible = (
                result.get("status") == CycleStatus.COMPLETED.value
                and verification.get("sequence_verified") is True
                and verification.get("omega_to_alpha_verified") is True
                and not result.get("contradictions")
            )
            if eligible:
                require_self_authority(self.self_authority, InternalAction.LEARN)
            self.brain.engine.cycle_id = max(
                self.brain.engine.cycle_id, int(brain["cycle_id"])
            )
            if eligible:
                self.brain.learn_verified_cycle()
            if verification.get("sequence_verified") is True:
                self.sequence = max(self.sequence, int(prediction["next_sequence"]))
            self._brain_result_rowid = rowid
            saved.update({
                "sequence": self.sequence,
                "last_cycle_id": cycle_id,
                "last_cycle_status": result["status"],
                "last_artifact_sha256": result.get("artifact_sha256"),
                "brain": brain,
                "brain_checkpoint": self.brain.checkpoint(),
                "brain_result_rowid": rowid,
                "contradictions": result.get("contradictions", []),
                # Recovery must not make an old cycle appear freshly executed.
                "updated_at": recorded_at,
            })
            changed = True
        if changed and persist:
            self._write_state(saved)

    def _speak_dialogue(self, dialogue: Mapping[str, Any]) -> None:
        """Print changed internal dialogue only when speech output is enabled."""
        if not self.speak or not dialogue:
            return

        rendered = json.dumps(dialogue, sort_keys=True)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if digest == self._last_spoken_digest:
            return

        self._last_spoken_digest = digest
        print(
            "GARVIS[Observer]: " + str(dialogue.get("observer", "")),
            flush=True,
        )
        print(
            "GARVIS[Skeptic]: " + str(dialogue.get("skeptic", "")),
            flush=True,
        )
        print(
            "GARVIS[Planner]: " + str(dialogue.get("planner", "")),
            flush=True,
        )

    def run_once(self) -> OABState:
        """Reconcile durable results and run one governed heartbeat cycle."""
        self._recover_brain_results()

        def observe() -> Mapping[str, Any]:
            """Capture the current sequence and read-only repository evidence."""
            snapshot = observe_system(
                self.repository_root,
                self.side_effects.pending_count(),
            )
            return {
                "sequence": self.sequence,
                "pending_protected_actions": (
                    self.side_effects.pending_count()
                ),
                "observed_at": time.time(),
                "system": snapshot.to_payload(),
            }

        def predict(pre: Mapping[str, Any]) -> Mapping[str, Any]:
            """Assess repository availability and predict the next internal sequence."""
            system = dict(pre.get("system", {}))
            brain = self.brain.assess(system)
            needs_attention = bool(
                not system.get("repository_available", False)
                or system.get("dirty_paths")
                or system.get("pending_protected_actions", 0)
                or (
                    system.get("branch")
                    and system.get("branch") != "main"
                )
            )
            return {
                "next_sequence": int(pre["sequence"]) + 1,
                "heartbeat_should_continue": True,
                "system_attention_expected": needs_attention,
                "expected_return_phase": "RECEIVE",
                "brain": brain,
            }

        def propose(
            pre: Mapping[str, Any],
            pred: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            """Prepare an internal observation with its frozen brain assessment."""
            snapshot = observe_system(
                self.repository_root,
                self.side_effects.pending_count(),
            )
            dialogue = build_internal_dialogue(snapshot)
            return {
                "kind": "internal_system_observation",
                "next_sequence": int(pred["next_sequence"]),
                "internal_dialogue": dialogue,
                "observed_system": dict(pre.get("system", {})),
                "brain": dict(pred["brain"]),
            }

        def plan(
            _pre: Mapping[str, Any],
            _pred: Mapping[str, Any],
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            """Describe phase progression without authorizing a repair."""
            dialogue = dict(proposal["internal_dialogue"])
            return {
                "steps": PHASE_NAMES + ("RECEIVE",),
                "proposal_kind": proposal["kind"],
                "repair_candidate": dialogue.get(
                    "repair_candidate",
                    "no_repair_needed",
                ),
                "capability_is_not_authorization": True,
            }

        def execute_internal(
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            """Advance the sequence only after the consolidation authority check."""
            require_self_authority(
                self.self_authority,
                InternalAction.CONSOLIDATE,
            )
            self.sequence = int(proposal["next_sequence"])
            next_phase = PHASE_NAMES[
                oab_wrap_phase_index(HEARTBEAT_PHASE_COUNT - 1)
            ]
            return {
                "sequence": self.sequence,
                "internal_dialogue": dict(
                    proposal["internal_dialogue"]
                ),
                "next_phase": next_phase,
                "brain": dict(proposal["brain"]),
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
            """Compare sequence and return phase with the frozen prediction."""
            expected = int(pred["next_sequence"])
            observed = int(post["sequence"])
            contradictions = []

            if expected != observed:
                contradictions.append(
                    {
                        "claim": "heartbeat sequence advanced as predicted",
                        "expected": expected,
                        "observed": observed,
                    }
                )

            if post.get("next_phase") != "RECEIVE":
                contradictions.append(
                    {
                        "claim": "omega closes directly into alpha",
                        "expected": "RECEIVE",
                        "observed": post.get("next_phase"),
                    }
                )

            return {
                "sequence_verified": expected == observed,
                "omega_to_alpha_verified": (
                    post.get("next_phase") == "RECEIVE"
                ),
                "phi_status": "HYPOTHESIS_UNDER_TEST",
                "contradictions": contradictions,
            }

        def learn(state: OABState) -> None:
            """Reconcile eligible committed results and persist the completed cycle state."""
            require_self_authority(
                self.self_authority,
                InternalAction.LEARN,
            )
            self._recover_brain_results(persist=False)
            self._persist_state(state)
            if isinstance(state.raw_post, Mapping):
                dialogue = state.raw_post.get("internal_dialogue", {})
                if isinstance(dialogue, Mapping):
                    self._speak_dialogue(dialogue)

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
                ),
                "returned_to_receive": (
                    post.get("next_phase") == "RECEIVE"
                ),
            },
            learn=learn,
            self_claims=(
                "I am GARVIS.",
                "My heartbeat service completed a software cycle.",
                "I am recording observations, predictions, and contradictions.",
            ),
            provenance={
                "system": "GARVIS",
                "service": "automatic-heartbeat-v2-smart",
                "creator": self.creator_authority.creator,
                "authority_source": (
                    self.creator_authority.authority_source
                ),
                "math_status": "HYPOTHESIS_UNDER_TEST",
                "dialogue_status": "INTERNAL_DIALOGUE_NOT_EVIDENCE",
            },
        )

    def run_forever(self) -> None:
        """Run heartbeat cycles with bounded backoff after failures."""
        backoff = max(0.1, self.interval_seconds or 0.1)

        while True:
            wait_seconds = self.interval_seconds
            try:
                state = self.run_once()
                if state.status is CycleStatus.FAILED:
                    self.rebound_count += 1
                    wait_seconds = backoff
                    backoff = min(30.0, backoff * 2.0)
                else:
                    backoff = max(
                        0.1,
                        self.interval_seconds or 0.1,
                    )
            except KeyboardInterrupt:
                raise
            except Exception:
                self.rebound_count += 1
                wait_seconds = backoff
                backoff = min(30.0, backoff * 2.0)

            # This cadence is represented as Alpha/RECEIVE state.
            # It is scheduler time, not a claim that cognition stopped.
            time.sleep(max(0.0, wait_seconds))

    def health(self) -> Mapping[str, Any]:
        """Report persisted cycle health with freshness and contradiction checks."""
        state = dict(self._read_state())
        updated_at = float(state.get("updated_at", 0.0) or 0.0)
        age_seconds = (
            None
            if updated_at <= 0.0
            else max(0.0, time.time() - updated_at)
        )
        freshness_window = max(
            5.0,
            self.interval_seconds * 3.0,
        )

        running = (
            age_seconds is not None
            and age_seconds <= freshness_window
        )
        healthy = (
            running
            and state.get("last_cycle_status")
            == CycleStatus.COMPLETED.value
            and not state.get("contradictions")
        )

        state["heartbeat_running"] = running
        state["heartbeat_healthy"] = healthy
        state["age_seconds"] = age_seconds
        state["freshness_window_seconds"] = freshness_window
        state["phi_status"] = "HYPOTHESIS_UNDER_TEST"
        state.setdefault("current_phase", "RECEIVE")
        state.setdefault(
            "alpha_omega_closure",
            "CONSOLIDATE->RECEIVE",
        )
        state.setdefault(
            "witness_capture",
            "automatic_non_blocking",
        )
        state.setdefault("rebound_count", self.rebound_count)
        return state

    def close(self) -> None:
        """Close the prediction and side-effect ledger connections."""
        self.predictions.close()
        self.side_effects.close()
