"""Connect the existing Hypercube Brain to local heartbeat observations.

Only verified software-cycle outcomes train this binding. A brain assessment
is advisory and does not authorize repository repairs or external actions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping

from hypercube_brain.core import ActionProposal, HypercubeBrainEngine, Observation


class HeartbeatBrainBinding:
    ACTION = "internal_system_observation"
    OUTCOME = "heartbeat_sequence_advanced_and_returned_to_receive"
    HISTORY_LIMIT = 89

    def __init__(self, checkpoint: Mapping[str, Any] | None = None) -> None:
        """Restore validated assessment and verified-outcome counters, if supplied."""
        self.engine = HypercubeBrainEngine()
        if checkpoint:
            if checkpoint.get("version") != 1:
                raise ValueError("unsupported heartbeat brain checkpoint")
            cycles = checkpoint["cycle_id"]
            successes = checkpoint["verified_cycles"]
            if (
                type(cycles) is not int or type(successes) is not int
                or not 0 <= successes <= cycles
            ):
                raise ValueError("invalid heartbeat brain counters")
            self.engine.cycle_id = cycles
            if successes:
                self.engine.effects[self.ACTION] = Counter({self.OUTCOME: successes})
                self.engine.plasticity[f"{self.ACTION}->{self.OUTCOME}"] = min(
                    1.0, 0.5 + 0.08 * successes
                )

    def assess(self, system: Mapping[str, Any]) -> dict[str, Any]:
        """Return an advisory assessment from a single repository evidence source."""
        # One Git observation is one source. Repeated heartbeats must not be
        # counted as independent corroboration or verification of the whole repo.
        available = system.get("repository_available") is True
        result = self.engine.heartbeat(
            claim="The configured local repository is available for observation.",
            observations=[Observation(
                content="repository_available=" + str(available),
                source="heartbeat.observe_system.git",
                independent_group="local_repository",
                contradicts=not available,
            )],
            action=ActionProposal(name=self.ACTION),
        )
        self._trim()
        payload = asdict(result)
        payload["truth_state"] = result.truth_state.value
        payload["action_decision"] = result.action_decision.value
        payload["warnings"] = list(result.warnings)
        payload["external_action_allowed"] = False
        payload["math_status"] = "HYPOTHESIS_UNDER_TEST"
        return payload

    def learn_verified_cycle(self) -> None:
        """Record one caller-verified internal outcome and bound memory history."""
        self.engine.learn_action_effect(
            action=self.ACTION, observed_outcome=self.OUTCOME, verified=True
        )
        self._trim()

    def _trim(self) -> None:
        """Keep only the configured number of recent episodes and events."""
        del self.engine.episodes[:-self.HISTORY_LIMIT]
        del self.engine.events[:-self.HISTORY_LIMIT]

    def checkpoint(self) -> dict[str, Any]:
        """Return versioned counters sufficient to reconstruct narrow outcome memory."""
        return {
            "version": 1,
            "cycle_id": self.engine.cycle_id,
            "verified_cycles": self.engine.effects.get(self.ACTION, {}).get(self.OUTCOME, 0),
        }
