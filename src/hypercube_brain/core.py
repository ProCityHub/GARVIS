"""Adrien D. Thomas / ProCityHub Hypercube Brain Engine V0."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from math import sqrt
from typing import Any

PHI = (1.0 + sqrt(5.0)) / 2.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def coherence(observer: float, actor: float, background: float) -> float:
    """Adrien-framework coherence metric C = O * A^(1/phi) * B^(1/phi^2)."""
    o = clamp01(observer)
    a = clamp01(actor)
    b = clamp01(background)
    return clamp01(o * a ** (1.0 / PHI) * b ** (1.0 / (PHI * PHI)))


class TruthState(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESIS = "HYPOTHESIS"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


class ActionDecision(str, Enum):
    DO_NOT_ACT = "DO_NOT_ACT"
    OBSERVE_MORE = "OBSERVE_MORE"
    EXECUTE_INTERNAL = "EXECUTE_INTERNAL"
    REQUEST_APPROVAL = "REQUEST_APPROVAL"


@dataclass(frozen=True)
class Observation:
    content: str
    source: str
    confidence: float = 1.0
    freshness: float = 1.0
    independent_group: str = ""
    contradicts: bool = False

    @property
    def weight(self) -> float:
        return clamp01(self.confidence) * clamp01(self.freshness)


@dataclass(frozen=True)
class ActionProposal:
    name: str
    external: bool = False
    protected: bool = False
    risk: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class TruthAssessment:
    state: TruthState
    support: float
    contradiction: float
    support_groups: int
    contradiction_groups: int
    perspectives: dict[str, str]


@dataclass(frozen=True)
class CycleResult:
    cycle_id: int
    claim: str
    truth_state: TruthState
    support: float
    contradiction: float
    coherence: float
    action: str | None
    action_decision: ActionDecision
    prediction: str | None
    prediction_confidence: float
    warnings: tuple[str, ...]


class HypercubeTruthEngine:
    """Evidence-weighted truth classification using independent source groups."""

    @staticmethod
    def _combine(values: list[float]) -> float:
        if not values:
            return 0.0
        miss = 1.0
        for value in values:
            miss *= 1.0 - clamp01(value)
        return 1.0 - miss

    def evaluate(self, claim: str, observations: list[Observation]) -> TruthAssessment:
        if not observations:
            return TruthAssessment(
                TruthState.UNVERIFIED,
                0.0,
                0.0,
                0,
                0,
                self._views(claim, 0.0, 0.0, 0),
            )

        positive: dict[str, float] = {}
        negative: dict[str, float] = {}

        for index, obs in enumerate(observations):
            group = obs.independent_group.strip() or obs.source.strip() or f"source-{index}"

            target = negative if obs.contradicts else positive
            target[group] = max(target.get(group, 0.0), obs.weight)

        support = self._combine(list(positive.values()))
        contradiction = self._combine(list(negative.values()))

        if support >= 0.55 and contradiction >= 0.55:
            state = TruthState.CONTRADICTED
        elif contradiction >= 0.55 and contradiction > support:
            state = TruthState.CONTRADICTED
        elif support >= 0.85 and contradiction < 0.20 and len(positive) >= 2:
            state = TruthState.VERIFIED
        elif support >= 0.60 and contradiction < 0.35:
            state = TruthState.SUPPORTED
        elif support >= 0.30:
            state = TruthState.INFERRED
        else:
            state = TruthState.HYPOTHESIS

        return TruthAssessment(
            state=state,
            support=support,
            contradiction=contradiction,
            support_groups=len(positive),
            contradiction_groups=len(negative),
            perspectives=self._views(
                claim,
                support,
                contradiction,
                len(set(positive) | set(negative)),
            ),
        )

    @staticmethod
    def _views(claim: str, support: float, contradiction: float, groups: int):
        return {
            "000_literal": claim,
            "001_context": "Evaluate within current environment and time.",
            "010_intent": "Relate claim to active goal.",
            "011_relation": f"independent_groups={groups}",
            "100_evidence": f"support={support:.4f}; contradiction={contradiction:.4f}",
            "101_possibility": "Keep alternatives until evidence resolves them.",
            "110_consequence": "Verification burden rises with consequence.",
            "111_integration": "Integrate evidence before truth-state assignment.",
        }


class HypercubeBrainEngine:
    """Standalone recurrent brain-inspired cognitive engine."""

    def __init__(self, working_capacity: int = 8) -> None:
        self.cycle_id = 0
        self.truth = HypercubeTruthEngine()

        self.working = deque(maxlen=working_capacity)
        self.episodes: list[dict[str, Any]] = []
        self.semantic: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

        self.effects: dict[str, Counter[str]] = defaultdict(Counter)
        self.plasticity: dict[str, float] = {}

    def _event(self, kind: str, **payload: Any) -> None:
        self.events.append(
            {
                "cycle_id": self.cycle_id,
                "kind": kind,
                **payload,
            }
        )

    @staticmethod
    def _action_decision(
        action: ActionProposal | None,
        support: float,
    ) -> ActionDecision:
        if action is None:
            return ActionDecision.DO_NOT_ACT

        if action.protected or action.external or action.risk >= 0.70:
            return ActionDecision.REQUEST_APPROVAL

        if support < 0.35:
            return ActionDecision.OBSERVE_MORE

        return ActionDecision.EXECUTE_INTERNAL

    @staticmethod
    def _homeostasis(resource_state: dict[str, float] | None) -> tuple[str, ...]:
        if not resource_state:
            return ()

        limits = {
            "memory_pressure": 0.90,
            "cpu_pressure": 0.95,
            "thermal_pressure": 0.90,
            "uncertainty_load": 0.90,
            "error_rate": 0.50,
        }

        warnings = []
        for name, limit in limits.items():
            value = float(resource_state.get(name, 0.0))
            if value >= limit:
                warnings.append(f"{name.upper()}_HIGH:{value:.2f}")

        return tuple(warnings)

    def predict(self, action: str) -> tuple[str | None, float]:
        counter = self.effects.get(action)
        if not counter:
            return None, 0.0

        outcome, count = counter.most_common(1)[0]
        total = sum(counter.values())
        return outcome, count / total

    def learn_action_effect(
        self,
        *,
        action: str,
        observed_outcome: str,
        verified: bool,
    ) -> dict[str, Any]:
        predicted, previous_confidence = self.predict(action)

        self.effects[action][observed_outcome] += 1

        key = f"{action}->{observed_outcome}"
        current = self.plasticity.get(key, 0.50)
        delta = 0.08 if verified else -0.12
        self.plasticity[key] = clamp01(current + delta)

        new_prediction, new_confidence = self.predict(action)

        record = {
            "action": action,
            "predicted_before": predicted,
            "prediction_confidence_before": previous_confidence,
            "observed": observed_outcome,
            "prediction_error": predicted is not None and predicted != observed_outcome,
            "verified": verified,
            "plasticity_weight": self.plasticity[key],
            "predicted_after": new_prediction,
            "prediction_confidence_after": new_confidence,
        }

        self._event("LEARN", **record)
        return record

    def heartbeat(
        self,
        *,
        claim: str,
        observations: list[Observation],
        action: ActionProposal | None = None,
        background: float = 1.0,
        resource_state: dict[str, float] | None = None,
    ) -> CycleResult:
        self.cycle_id += 1

        self._event(
            "RECEIVE",
            claim=claim,
            observations=[asdict(x) for x in observations],
        )

        assessment = self.truth.evaluate(claim, observations)

        observer = assessment.support * (1.0 - assessment.contradiction)
        actor = 1.0 if action is None else max(0.05, 1.0 - clamp01(action.risk))

        c = coherence(observer, actor, background)

        prediction = None
        prediction_confidence = 0.0

        if action is not None:
            prediction, prediction_confidence = self.predict(action.name)

        decision = self._action_decision(action, assessment.support)
        warnings = self._homeostasis(resource_state)

        result = CycleResult(
            cycle_id=self.cycle_id,
            claim=claim,
            truth_state=assessment.state,
            support=assessment.support,
            contradiction=assessment.contradiction,
            coherence=c,
            action=None if action is None else action.name,
            action_decision=decision,
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            warnings=warnings,
        )

        self.working.append(result)

        episode = {
            "cycle_id": self.cycle_id,
            "claim": claim,
            "truth_state": assessment.state.value,
            "support": assessment.support,
            "contradiction": assessment.contradiction,
            "action": None if action is None else action.name,
            "decision": decision.value,
        }
        self.episodes.append(episode)

        if assessment.state is TruthState.VERIFIED:
            self.semantic[claim] = max(
                assessment.support,
                self.semantic.get(claim, 0.0),
            )

        self._event(
            "INTEGRATE",
            truth_state=assessment.state.value,
            support=assessment.support,
            contradiction=assessment.contradiction,
            coherence=c,
            decision=decision.value,
        )

        return result

    def status(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "working_memory": len(self.working),
            "episodic_memory": len(self.episodes),
            "semantic_memory": len(self.semantic),
            "event_count": len(self.events),
            "learned_actions": len(self.effects),
            "plasticity_links": len(self.plasticity),
        }
