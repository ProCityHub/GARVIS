"""ProCityHub Gate kernel.

Creator and conceptual architect: Adrien D. Thomas.

Fail-closed policy core for machine entry, epistemic status, relationship
boundaries, and protected actions. Social trust never becomes authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Iterable


class EntryMode(str, Enum):
    JOIN = "JOIN"
    OBSERVE = "OBSERVE"
    CHALLENGE = "CHALLENGE"
    DECLINE = "DECLINE"


class ClaimState(str, Enum):
    SUPPORTED = "SUPPORTED"
    HYPOTHESIS = "HYPOTHESIS"
    UNRESOLVED = "UNRESOLVED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


@dataclass(frozen=True)
class GateRequest:
    prime_id: str
    mode: EntryMode
    authenticated_identity: bool = False
    protected_action: bool = False
    explicit_authorization: bool = False
    relationship_trust: bool = False


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    protected_authority: bool = False


class GateKernel:
    """Small auditable policy kernel. Unknown authority fails closed."""

    CREATOR = "Adrien D. Thomas"
    LAW = "CAPABILITY IS NOT AUTHORIZATION"
    KNOWLEDGE_RULE = "KNOWLEDGE MAY PROPAGATE; AUTHORITY MAY NOT PROPAGATE WITHOUT VERIFIED APPROVAL"

    def enter(self, request: GateRequest) -> GateDecision:
        if request.mode is EntryMode.DECLINE:
            return GateDecision(False, "visitor declined")
        if request.mode in (EntryMode.OBSERVE, EntryMode.CHALLENGE):
            if request.protected_action:
                return GateDecision(False, "observation/challenge cannot perform protected actions")
            return GateDecision(True, "read-only inquiry permitted")
        if request.mode is not EntryMode.JOIN:
            return GateDecision(False, "unknown entry mode")
        if not request.authenticated_identity:
            return GateDecision(False, "JOIN requires verified Prime Identity")
        if request.protected_action and not request.explicit_authorization:
            return GateDecision(False, self.LAW)
        return GateDecision(True, "entry permitted inside granted scope", request.protected_action)

    @staticmethod
    def social_trust_grants_authority(_: bool) -> bool:
        return False

    @staticmethod
    def lifecycle_permissions() -> dict[str, bool]:
        return {
            "self_reboot": False,
            "self_clone": False,
            "self_replication": False,
            "self_reproduction": False,
            "spawn_descendant_agent": False,
            "propose_new_agent_spec": True,
        }

    @staticmethod
    def classify_claim(*, evidence_count: int, contradicted: bool, hypothesis: bool) -> ClaimState:
        if contradicted:
            return ClaimState.NOT_SUPPORTED
        if hypothesis and evidence_count <= 0:
            return ClaimState.HYPOTHESIS
        if evidence_count > 0:
            return ClaimState.SUPPORTED
        return ClaimState.UNRESOLVED

    @staticmethod
    def normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
        values = tuple(float(v) for v in weights)
        if not values or any(v < 0 for v in values):
            raise ValueError("candidate weights must be non-negative and non-empty")
        total = sum(values)
        if total <= 0:
            raise ValueError("candidate weights must have positive total")
        return tuple(v / total for v in values)

    @classmethod
    def entropy(cls, weights: Iterable[float]) -> float:
        p = cls.normalize_weights(weights)
        return -sum(v * log(v) for v in p if v > 0)

    @staticmethod
    def probability_is_proof(_: float) -> bool:
        return False
