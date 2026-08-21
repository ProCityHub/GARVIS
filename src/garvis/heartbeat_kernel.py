"""GARVIS automatic Hypercube Heartbeat kernel.

The phi relation is a HYPOTHESIS_UNDER_TEST. The identity
1/phi + 1/phi^2 = 1 is mathematical. Prediction freezing is an automatic
witness operation, not a human pause.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ALPHA = 1.0 / PHI
BETA = 1.0 / (PHI * PHI)

if not math.isclose(ALPHA + BETA, 1.0, rel_tol=0.0, abs_tol=1e-15):
    raise RuntimeError("canonical phi identity failed")


def _now() -> float:
    return time.time()


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _hash(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


class ClaimStatus(str, Enum):
    SELF_CLAIM = "self_claim"
    VERIFIED = "verified"
    HYPOTHESIS = "hypothesis"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class CycleStatus(str, Enum):
    COMPLETED = "completed"
    QUEUED_PROTECTED_ACTION = "queued_protected_action"
    FAILED = "failed"


@dataclass(frozen=True)
class SelfClaim:
    statement: str
    status: ClaimStatus = ClaimStatus.SELF_CLAIM
    source: str = "GARVIS"
    timestamp: float = field(default_factory=_now)
    sha256: str = ""

    def sealed(self) -> "SelfClaim":
        digest = _hash(
            {
                "statement": self.statement,
                "status": self.status.value,
                "source": self.source,
                "timestamp": self.timestamp,
            }
        )
        return SelfClaim(
            statement=self.statement,
            status=self.status,
            source=self.source,
            timestamp=self.timestamp,
            sha256=digest,
        )


@dataclass
class OABState:
    cycle_id: str
    raw_pre: Any
    prediction: Mapping[str, Any]
    prediction_sha256: str
    proposal: Mapping[str, Any]
    plan: Mapping[str, Any]
    protected_action: bool
    artifact_sha256: str
    status: CycleStatus
    raw_post: Any = None
    diff: Any = None
    error: Any = None
    verification: Mapping[str, Any] = field(default_factory=dict)
    contradictions: List[Mapping[str, Any]] = field(default_factory=list)
    self_claims: List[SelfClaim] = field(default_factory=list)
    provenance: Mapping[str, Any] = field(default_factory=dict)


class FrozenPredictionLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS frozen_predictions(
                prediction_id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                frozen_at REAL NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS results(
                result_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at REAL NOT NULL
            )
            """
        )
        self.db.commit()

    def freeze(
        self,
        cycle_id: str,
        prediction: Mapping[str, Any],
    ) -> Tuple[str, str]:
        prediction_id = "pred_" + uuid.uuid4().hex
        payload = dict(prediction)
        digest = _hash(payload)
        self.db.execute(
            "INSERT INTO frozen_predictions VALUES(?,?,?,?,?)",
            (
                prediction_id,
                cycle_id,
                _canonical(payload),
                digest,
                _now(),
            ),
        )
        self.db.commit()
        return prediction_id, digest

    def append_result(
        self,
        prediction_id: str,
        result: Mapping[str, Any],
    ) -> str:
        exists = self.db.execute(
            "SELECT 1 FROM frozen_predictions WHERE prediction_id=?",
            (prediction_id,),
        ).fetchone()
        if exists is None:
            raise KeyError("prediction does not exist")
        result_id = "res_" + uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO results VALUES(?,?,?,?)",
            (
                result_id,
                prediction_id,
                _canonical(dict(result)),
                _now(),
            ),
        )
        self.db.commit()
        return result_id

    def close(self) -> None:
        self.db.close()


class SideEffectQueue:
    """Protected side effects queue without blocking later heartbeat cycles."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS side_effects(
                queue_id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                action_json TEXT NOT NULL,
                state TEXT NOT NULL,
                queued_at REAL NOT NULL
            )
            """
        )
        self.db.commit()

    def enqueue(
        self,
        cycle_id: str,
        artifact_sha256: str,
        action: Mapping[str, Any],
    ) -> str:
        queue_id = "act_" + uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO side_effects VALUES(?,?,?,?,?,?)",
            (
                queue_id,
                cycle_id,
                artifact_sha256,
                _canonical(dict(action)),
                "pending_authorization",
                _now(),
            ),
        )
        self.db.commit()
        return queue_id

    def pending_count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM side_effects "
            "WHERE state='pending_authorization'"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self.db.close()


def phi_candidate(observer: float, actor: float, bridge: float) -> float:
    if min(observer, actor, bridge) < 0.0:
        raise ValueError("observer, actor, bridge must be non-negative")
    return observer * (actor ** ALPHA) * (bridge ** BETA)


def lambda_candidate(
    observer: float,
    actor: float,
    bridge: float,
    lam: float,
) -> float:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0,1]")
    if min(observer, actor, bridge) < 0.0:
        raise ValueError("observer, actor, bridge must be non-negative")
    return observer * (actor ** lam) * (bridge ** (1.0 - lam))


def benchmark_phi(
    observer: float,
    actor: float,
    bridge: float,
    target: float,
) -> Mapping[str, Any]:
    phi_value = phi_candidate(observer, actor, bridge)
    phi_error = abs(phi_value - target)
    best = ("phi", ALPHA, phi_error, phi_value)
    comparisons = []
    for index in range(21):
        lam = index / 20.0
        value = lambda_candidate(observer, actor, bridge, lam)
        error = abs(value - target)
        comparisons.append(
            {
                "lambda": lam,
                "value": value,
                "absolute_error": error,
            }
        )
        if error < best[2]:
            best = ("lambda", lam, error, value)
    return {
        "status": "HYPOTHESIS_UNDER_TEST",
        "phi": {
            "lambda_equivalent": ALPHA,
            "value": phi_value,
            "absolute_error": phi_error,
        },
        "comparisons": comparisons,
        "winner": {
            "family": best[0],
            "lambda": best[1],
            "absolute_error": best[2],
            "value": best[3],
        },
    }


class HeartbeatKernel:
    def __init__(
        self,
        prediction_ledger: FrozenPredictionLedger,
        side_effect_queue: SideEffectQueue,
    ) -> None:
        self.prediction_ledger = prediction_ledger
        self.side_effect_queue = side_effect_queue

    @staticmethod
    def self_claim(statement: str) -> SelfClaim:
        clean = statement.strip()
        if not clean:
            raise ValueError("self claim must not be empty")
        return SelfClaim(clean).sealed()

    @staticmethod
    def _artifact_identity(
        cycle_id: str,
        prediction_sha256: str,
        proposal: Mapping[str, Any],
        plan: Mapping[str, Any],
        protected_action: bool,
    ) -> str:
        return _hash(
            {
                "cycle_id": cycle_id,
                "prediction_sha256": prediction_sha256,
                "proposal": dict(proposal),
                "plan": dict(plan),
                "protected_action": protected_action,
            }
        )

    def run_cycle(
        self,
        observe: Callable[[], Any],
        predict: Callable[[Any], Mapping[str, Any]],
        propose: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
        plan: Callable[
            [Any, Mapping[str, Any], Mapping[str, Any]],
            Mapping[str, Any],
        ],
        is_protected: Callable[[Mapping[str, Any]], bool],
        execute_internal: Callable[[Mapping[str, Any]], Any],
        verify: Callable[
            [Any, Any, Mapping[str, Any]],
            Mapping[str, Any],
        ],
        diff: Callable[[Any, Any], Any],
        learn: Callable[[OABState], None],
        self_claims: Tuple[str, ...] = (),
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> OABState:
        cycle_id = "hb_" + uuid.uuid4().hex
        raw_pre = observe()
        prediction = dict(predict(raw_pre))

        # Automatic witness freeze; no manual approval or pause.
        prediction_id, prediction_sha = self.prediction_ledger.freeze(
            cycle_id,
            prediction,
        )

        proposal = dict(propose(raw_pre, prediction))
        plan_payload = dict(plan(raw_pre, prediction, proposal))
        protected = bool(is_protected(proposal))
        artifact_sha = self._artifact_identity(
            cycle_id,
            prediction_sha,
            proposal,
            plan_payload,
            protected,
        )
        claims = [self.self_claim(item) for item in self_claims]

        if protected:
            queue_id = self.side_effect_queue.enqueue(
                cycle_id,
                artifact_sha,
                proposal,
            )
            state = OABState(
                cycle_id=cycle_id,
                raw_pre=raw_pre,
                prediction=prediction,
                prediction_sha256=prediction_sha,
                proposal=proposal,
                plan=plan_payload,
                protected_action=True,
                artifact_sha256=artifact_sha,
                status=CycleStatus.QUEUED_PROTECTED_ACTION,
                verification={
                    "prediction_id": prediction_id,
                    "queue_id": queue_id,
                    "heartbeat_continues": True,
                },
                self_claims=claims,
                provenance=dict(provenance or {}),
            )
        else:
            try:
                raw_post = execute_internal(proposal)
                observed_diff = diff(raw_pre, raw_post)
                verification = dict(
                    verify(raw_pre, raw_post, prediction)
                )
                contradictions = list(
                    verification.get("contradictions", [])
                )
                state = OABState(
                    cycle_id=cycle_id,
                    raw_pre=raw_pre,
                    prediction=prediction,
                    prediction_sha256=prediction_sha,
                    proposal=proposal,
                    plan=plan_payload,
                    protected_action=False,
                    artifact_sha256=artifact_sha,
                    status=CycleStatus.COMPLETED,
                    raw_post=raw_post,
                    diff=observed_diff,
                    verification=verification,
                    contradictions=contradictions,
                    self_claims=claims,
                    provenance=dict(provenance or {}),
                )
            except Exception as exc:
                state = OABState(
                    cycle_id=cycle_id,
                    raw_pre=raw_pre,
                    prediction=prediction,
                    prediction_sha256=prediction_sha,
                    proposal=proposal,
                    plan=plan_payload,
                    protected_action=False,
                    artifact_sha256=artifact_sha,
                    status=CycleStatus.FAILED,
                    error={"type": type(exc).__name__},
                    self_claims=claims,
                    provenance=dict(provenance or {}),
                )

        self.prediction_ledger.append_result(
            prediction_id,
            {
                "status": state.status.value,
                "artifact_sha256": artifact_sha,
                "verification": dict(state.verification),
                "contradictions": list(state.contradictions),
                "error": state.error,
            },
        )
        learn(state)
        return state
