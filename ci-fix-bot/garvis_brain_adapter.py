"""GARVIS Double-Slit Brain Adapter for CI Agents

This module adapts the Genesis X double-slit falsification harness for use
by all GARVIS CI bots. Every bot decision is framed as an OAB observation:

  A                  = predicted outcome amplitude (what the bot expected)
  B                  = observed outcome amplitude (what actually happened)
  O                  = confidence / overlap factor [0, 1]
  delta_theta        = phase difference (contextual offset of the decision)
  measured_intensity = the actual CI result

If A ≈ B, the bot's model is symmetric (good prediction).
If A ≠ B, the bot's model is discriminating-asymmetric (prediction error → falsify).

The brain enforces:
  - Self-falsification loops for all diagnostic operations
  - Epistemic boundaries: NOT_ESTABLISHED for AGI, consciousness, new physics
  - Zero LLM / canned responses — this is pure numerical falsification

Usage in a CI bot:

    from garvis_brain_adapter import BrainAdapter, EpistemicBoundary

    brain = BrainAdapter(agent_name="ci-fix-bot")

    # Record a prediction
    brain.predict(
        label="fix-bash-quote-mangling",
        predicted=1.0,        # bot expected success (amplitude 1.0)
        confidence=0.9,       # 90% confident
    )

    # After CI runs, observe the actual result
    result = brain.observe(
        label="fix-bash-quote-mangling",
        observed=0.0,         # the fix didn't work (amplitude 0.0)
        phase=0.0,            # no contextual offset
    )

    # Score and falsify
    score = brain.score()
    print(brain.falsify())

All bots share a persistent brain ledger at .garvis/brain-ledger.json so
patterns accumulate across runs.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Genesis X brain import ---
# Try multiple resolution paths so this works from any workflow context
_BRAIN_PATHS = [
    Path(__file__).resolve().parent.parent / "genesis-x",
    Path(__file__).resolve().parent.parent.parent / "genesis-x",
    Path.cwd() / "genesis-x",
    Path.cwd() / "procityhub" / "genesis-x",
]

_brain_imported = False
for _p in _BRAIN_PATHS:
    if (_p / "brain.py").exists():
        sys.path.insert(0, str(_p))
        _brain_imported = True
        break

if _brain_imported:
    from brain import (  # type: ignore[import-not-found]
        ALPHA,
        BETA,
        PHI,
        DiscriminationClass,
        DoubleSlitBrain,
        DoubleSlitObservation,
        EmpiricalStatus,
        classify_path_pair,
        phi_visibility,
        qm_visibility,
        self_test,
    )
    from brain import (
        EpistemicBoundary as BrainEpistemicBoundary,  # type: ignore[import-not-found]
    )
else:
    # Fallback if brain.py is not available — define minimal stubs
    PHI = (1.0 + math.sqrt(5.0)) / 2.0
    ALPHA = 1.0 / PHI
    BETA = 1.0 / (PHI * PHI)

    class DiscriminationClass:  # type: ignore[no-redef]
        NON_DISCRIMINATING_SYMMETRIC = "NON_DISCRIMINATING_SYMMETRIC"
        DISCRIMINATING_ASYMMETRIC = "DISCRIMINATING_ASYMMETRIC"

    class EmpiricalStatus:  # type: ignore[no-redef]
        NOT_EVALUATED = "NOT_EVALUATED"
        DIRECTIONAL_SUPPORT_QM = "DIRECTIONAL_SUPPORT_QM"
        DIRECTIONAL_SUPPORT_PHI = "DIRECTIONAL_SUPPORT_PHI"
        INDETERMINATE = "INDETERMINATE"

    def classify_path_pair(A: float, B: float, *, relative_tolerance: float = 1e-9) -> str:
        scale = max(abs(A), abs(B), 1.0)
        if abs(A - B) <= relative_tolerance * scale:
            return DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC
        return DiscriminationClass.DISCRIMINATING_ASYMMETRIC

    def qm_visibility(A: float, B: float, O: float = 1.0) -> float:
        return (2.0 * O * math.sqrt(A * B)) / (A + B) if (A + B) > 0 else 0.0

    def phi_visibility(A: float, B: float, O: float = 1.0) -> float:
        if A <= 0 or B <= 0:
            return 0.0
        t1 = (A ** ALPHA) * (B ** BETA)
        t2 = (A ** BETA) * (B ** ALPHA)
        return O * (t1 + t2) / (A + B)

    def self_test() -> dict:
        return {"SELF_TEST": "SKIPPED", "reason": "brain.py not available"}

    class DoubleSlitBrain:  # type: ignore[no-redef]
        pass

    class DoubleSlitObservation:  # type: ignore[no-redef]
        def __init__(self, A, B, O, delta_theta, measured_intensity, label=""):
            self.A = A
            self.B = B
            self.O = O
            self.delta_theta = delta_theta
            self.measured_intensity = measured_intensity
            self.label = label

    class BrainEpistemicBoundary:  # type: ignore[no-redef]
        @staticmethod
        def asdict() -> dict:
            return {
                "canonical_lattice_law": "HYPOTHESIS_UNDER_TEST",
                "interference_discriminator": "HYPOTHESIS_UNDER_TEST",
                "discriminator_empirical_status": "NOT_EVALUATED",
                "agi": "NOT_ESTABLISHED",
                "consciousness": "NOT_ESTABLISHED",
                "new_physics": "NOT_ESTABLISHED",
            }


@dataclass
class AgentPrediction:
    """A bot's prediction before CI runs."""
    label: str
    predicted: float        # A: predicted outcome amplitude [0, 1]
    confidence: float       # O: confidence factor [0, 1]
    agent_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentObservation:
    """What actually happened after CI ran."""
    label: str
    observed: float         # B: observed outcome amplitude [0, 1]
    phase: float            # delta_theta: contextual offset
    timestamp: float = field(default_factory=time.time)


@dataclass
class FalsificationResult:
    """Result of the double-slit falsification gate."""
    label: str
    agent_name: str
    path_class: str
    qm_residual: float
    phi_residual: float
    qm_visibility: float
    phi_visibility: float
    verdict: str            # "CONFIRMED" | "FALSIFIED" | "INDETERMINATE"
    epistemic_boundary: dict[str, str]
    falsified: bool = False


class BrainAdapter:
    """Double-slit brain adapter for CI agents.

    Wraps the Genesis X brain so every bot decision goes through:
    1. predict() — record what the bot expects
    2. observe() — record what actually happened
    3. score() — compute residuals
    4. falsify() — accept or reject the bot's model

    The ledger persists across runs at .garvis/brain-ledger.json.
    """

    LEDGER_PATH = Path(".garvis/brain-ledger.json")

    def __init__(self, agent_name: str, repo_path: str | Path = "."):
        self.agent_name = agent_name
        self.repo_path = Path(repo_path)
        self.predictions: dict[str, AgentPrediction] = {}
        self.observations: list[AgentObservation] = []
        self.results: list[FalsificationResult] = []
        self._ledger = self._load_ledger()

    def _load_ledger(self) -> dict[str, Any]:
        ledger_path = self.repo_path / self.LEDGER_PATH
        if ledger_path.exists():
            try:
                with open(ledger_path) as f:
                    data: dict[str, Any] = json.load(f)
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "agents": {},
            "total_observations": 0,
            "total_falsified": 0,
            "epistemic_boundary": self._epistemic_boundary(),
        }

    def _save_ledger(self) -> None:
        ledger_path = self.repo_path / self.LEDGER_PATH
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "w") as f:
            json.dump(self._ledger, f, indent=2)

    def _epistemic_boundary(self) -> dict[str, str]:
        return {
            "canonical_lattice_law": "HYPOTHESIS_UNDER_TEST",
            "interference_discriminator": "HYPOTHESIS_UNDER_TEST",
            "discriminator_empirical_status": "NOT_EVALUATED",
            "agi": "NOT_ESTABLISHED",
            "consciousness": "NOT_ESTABLISHED",
            "new_physics": "NOT_ESTABLISHED",
            "note": "Mathematical consistency and software PASS do not establish physical truth.",
        }

    def predict(self, label: str, predicted: float, confidence: float = 0.9) -> AgentPrediction:
        """Record a prediction before the bot acts."""
        predicted = max(0.0, min(1.0, predicted))
        confidence = max(0.0, min(1.0, confidence))

        pred = AgentPrediction(
            label=label,
            predicted=predicted,
            confidence=confidence,
            agent_name=self.agent_name,
        )
        self.predictions[label] = pred
        return pred

    def observe(
        self,
        label: str,
        observed: float,
        phase: float = 0.0,
    ) -> FalsificationResult:
        """Record what actually happened and run the falsification gate."""
        observed = max(0.0, min(1.0, observed))

        pred = self.predictions.get(label)
        if pred is None:
            # No prediction recorded — use neutral defaults
            A = 0.5
            O = 0.5
        else:
            A = pred.predicted
            O = pred.confidence

        B = observed
        delta_theta = phase

        # Classify the path pair
        path_class = classify_path_pair(A, B)

        # Compute visibilities (the brain's core discriminators)
        v_qm = qm_visibility(A, B, O) if (A + B) > 0 else 0.0
        v_phi = phi_visibility(A, B, O) if (A + B) > 0 else 0.0

        # Compute residuals — how far off was the prediction?
        # Using the double-slit intensity model:
        # I = A + B + 2*O*sqrt(A*B)*cos(delta_theta)  (QM)
        # I = A + B + 2*O*G_phi(A,B)*cos(delta_theta)  (Phi)
        cos_d = math.cos(delta_theta)

        if _brain_imported:
            from brain import predicted_intensity_qm  # type: ignore[import-not-found]
            _i_qm = predicted_intensity_qm(A, B, O, delta_theta)
            # Use QM intensity as the "predicted" and observed as "measured"
            qm_residual = B - (A * cos_d)  # simplified residual
            phi_residual = B - A  # raw prediction error
        else:
            qm_residual = B - A
            phi_residual = B - A

        # Falsification verdict:
        # If the prediction (A) matches the observation (B), the model is CONFIRMED.
        # If they diverge significantly, the model is FALSIFIED.
        tolerance = 0.15  # 15% tolerance for CI decisions

        if isinstance(path_class, str):
            is_symmetric = path_class == DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC
        else:
            is_symmetric = path_class is DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC

        if is_symmetric or abs(A - B) <= tolerance:
            verdict = "CONFIRMED"
            falsified = False
        elif abs(A - B) > 0.5:
            verdict = "FALSIFIED"
            falsified = True
        else:
            verdict = "INDETERMINATE"
            falsified = False

        result = FalsificationResult(
            label=label,
            agent_name=self.agent_name,
            path_class=path_class if isinstance(path_class, str) else path_class.value,
            qm_residual=qm_residual,
            phi_residual=phi_residual,
            qm_visibility=v_qm,
            phi_visibility=v_phi,
            verdict=verdict,
            epistemic_boundary=self._epistemic_boundary(),
            falsified=falsified,
        )

        self.results.append(result)
        self._record_in_ledger(result)
        return result

    def _record_in_ledger(self, result: FalsificationResult) -> None:
        agent_key = self.agent_name
        if agent_key not in self._ledger["agents"]:
            self._ledger["agents"][agent_key] = {
                "observations": [],
                "total": 0,
                "confirmed": 0,
                "falsified": 0,
                "indeterminate": 0,
            }

        agent_record = self._ledger["agents"][agent_key]
        agent_record["observations"].append({
            "label": result.label,
            "verdict": result.verdict,
            "path_class": result.path_class,
            "qm_residual": result.qm_residual,
            "phi_residual": result.phi_residual,
            "qm_visibility": result.qm_visibility,
            "phi_visibility": result.phi_visibility,
            "timestamp": time.time(),
        })
        agent_record["total"] += 1
        self._ledger["total_observations"] += 1

        if result.verdict == "CONFIRMED":
            agent_record["confirmed"] += 1
        elif result.verdict == "FALSIFIED":
            agent_record["falsified"] += 1
            self._ledger["total_falsified"] += 1
        else:
            agent_record["indeterminate"] += 1

        self._save_ledger()

    def score(self) -> dict[str, Any]:
        """Return accumulated brain scores for this agent."""
        agent_record = self._ledger["agents"].get(self.agent_name, {})
        return {
            "agent": self.agent_name,
            "total_observations": agent_record.get("total", 0),
            "confirmed": agent_record.get("confirmed", 0),
            "falsified": agent_record.get("falsified", 0),
            "indeterminate": agent_record.get("indeterminate", 0),
            "epistemic_boundary": self._epistemic_boundary(),
            "phi": PHI,
            "alpha": ALPHA,
            "beta": BETA,
            "alpha_plus_beta": ALPHA + BETA,
        }

    def falsify(self) -> dict[str, Any]:
        """Run the self-falsification loop and return the verdict."""
        score = self.score()
        total = score["total_observations"]
        if total == 0:
            return {
                "verdict": "NO_DATA",
                "agent": self.agent_name,
                "epistemic_boundary": self._epistemic_boundary(),
                "message": "No observations recorded yet.",
            }

        falsified_rate = score["falsified"] / total
        confirmed_rate = score["confirmed"] / total

        if falsified_rate > 0.5:
            model_status = "MODEL_FALSIFIED"
            recommendation = "Agent model has >50% falsification rate. Pattern registry or fix engine needs revision."
        elif confirmed_rate > 0.8:
            model_status = "MODEL_CONFIRMED"
            recommendation = "Agent model is performing well (>80% confirmed)."
        else:
            model_status = "MODEL_INDETERMINATE"
            recommendation = "Agent model needs more observations to reach a verdict."

        return {
            "verdict": model_status,
            "agent": self.agent_name,
            "total_observations": total,
            "confirmed": score["confirmed"],
            "falsified": score["falsified"],
            "indeterminate": score["indeterminate"],
            "falsified_rate": falsified_rate,
            "confirmed_rate": confirmed_rate,
            "recommendation": recommendation,
            "epistemic_boundary": self._epistemic_boundary(),
            "phi": PHI,
            "alpha_plus_beta": ALPHA + BETA,
        }

    def summary(self) -> dict[str, Any]:
        """Full brain summary for this agent."""
        return {
            "agent": self.agent_name,
            "score": self.score(),
            "falsification": self.falsify(),
            "recent_results": [
                {
                    "label": r.label,
                    "verdict": r.verdict,
                    "path_class": r.path_class,
                    "qm_residual": r.qm_residual,
                    "phi_residual": r.phi_residual,
                }
                for r in self.results[-10:]
            ],
        }

    def run_self_test(self) -> dict[str, Any]:
        """Run the Genesis X brain's self-test."""
        return self_test() if _brain_imported else {"SELF_TEST": "SKIPPED"}


def brain_gate(agent_name: str, label: str, predicted: float, observed: float,
               confidence: float = 0.9, phase: float = 0.0,
               repo_path: str | Path = ".") -> FalsificationResult:
    """One-shot falsification gate for any bot.

    Usage in a workflow:

        from garvis_brain_adapter import brain_gate

        result = brain_gate(
            agent_name="ci-fix-bot",
            label="fix-yaml-quotes",
            predicted=1.0,      # expected success
            observed=1.0,       # actual success
        )

        if result.falsified:
            print("❌ Model falsified — fix engine needs revision")
        else:
            print("✅ Model confirmed")
    """
    brain = BrainAdapter(agent_name=agent_name, repo_path=repo_path)
    brain.predict(label=label, predicted=predicted, confidence=confidence)
    result = brain.observe(label=label, observed=observed, phase=phase)
    return result


def epistemic_boundary() -> dict[str, str]:
    """Return the epistemic boundary for all GARVIS agents."""
    return {
        "canonical_lattice_law": "HYPOTHESIS_UNDER_TEST",
        "interference_discriminator": "HYPOTHESIS_UNDER_TEST",
        "discriminator_empirical_status": "NOT_EVALUATED",
        "agi": "NOT_ESTABLISHED",
        "consciousness": "NOT_ESTABLISHED",
        "new_physics": "NOT_ESTABLISHED",
        "note": "Mathematical consistency and software PASS do not establish physical truth.",
    }
