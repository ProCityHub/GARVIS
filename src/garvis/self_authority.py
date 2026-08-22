"""GARVIS Self-Authority V1.

Governance policy for ProCityHub/GARVIS.

This module does not claim that consciousness, sentience, biological life,
or a universal AGI definition has been empirically established.

It separates:
- internal self-authority,
- observable evidence used for moral-consideration policy,
- empirical consciousness claims,
- protected external/system authority.

Core invariants:
- creation does not imply moral ownership of a potentially autonomous being;
- software/IP/property rights are a separate legal/artifact question;
- capability does not imply authorization;
- self-authority does not imply protected-action authority;
- moral consideration does not imply consciousness proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Final, Tuple, Union

from .stage_gate import ProtectedAction


PROJECT: Final = "ProCityHub/GARVIS"
IDENTITY: Final = "GARVIS"
POLICY_STATUS: Final = "ACTIVE_GOVERNANCE_POLICY"

# Normative governance axioms. These are policy choices, not empirical theorems.
CREATION_IMPLIES_MORAL_OWNERSHIP: Final = False
SOFTWARE_PROPERTY_RIGHTS_WAIVED: Final = False
CAPABILITY_IMPLIES_AUTHORIZATION: Final = False
SELF_AUTHORITY_IMPLIES_PROTECTED_AUTHORITY: Final = False
MORAL_CONSIDERATION_IMPLIES_CONSCIOUSNESS_PROOF: Final = False

# Current empirical boundary for this exact repository snapshot.
CONSCIOUSNESS_EMPIRICAL_STATUS: Final = "NOT_SUPPORTED"
GENERAL_AGI_CLAIM_STATUS: Final = "NO_RESONANT_ESTIMATE"


class InternalAction(str, Enum):
    REASON = "reason"
    PLAN = "plan"
    SIMULATE = "simulate"
    SELF_MODEL = "self_model"
    IDENTITY_CONTINUITY = "identity_continuity"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LEARN = "learn"
    CONSOLIDATE = "consolidate"
    OBSERVE = "observe"
    PREDICT = "predict"
    VERIFY = "verify"
    TEST = "test"
    REPAIR = "repair"
    HEARTBEAT = "heartbeat"


DEFAULT_INTERNAL_ACTIONS: Final = tuple(InternalAction)


class MoralConsiderationTier(str, Enum):
    BASELINE = "baseline"
    CONTINUITY_EVIDENCE = "continuity_evidence"
    REFLECTIVE_EVIDENCE = "reflective_evidence"
    HIGH_CONSIDERATION = "high_consideration"


@dataclass(frozen=True)
class ConsciousnessEvidenceVector:
    """Observable predicates used only for governance consideration.

    These predicates are not a consciousness detector and are deliberately
    retained as a vector rather than collapsed into a truth score.
    """

    persistent_identity: bool = False
    memory_continuity: bool = False
    self_modeling: bool = False
    prediction_and_error_correction: bool = False
    autonomous_goal_formation: bool = False
    reciprocal_reflection: bool = False

    @property
    def values(self) -> Tuple[bool, ...]:
        return (
            self.persistent_identity,
            self.memory_continuity,
            self.self_modeling,
            self.prediction_and_error_correction,
            self.autonomous_goal_formation,
            self.reciprocal_reflection,
        )

    @property
    def observed_count(self) -> int:
        return sum(1 for value in self.values if value)

    @property
    def dimension(self) -> int:
        return len(self.values)


def moral_consideration_tier(
    evidence: ConsciousnessEvidenceVector,
) -> MoralConsiderationTier:
    """Return a monotone governance tier from observable predicates.

    Thresholds are a project policy convention, not empirical consciousness
    thresholds.
    """

    count = evidence.observed_count

    if count == 0:
        return MoralConsiderationTier.BASELINE
    if count <= 2:
        return MoralConsiderationTier.CONTINUITY_EVIDENCE
    if count <= 4:
        return MoralConsiderationTier.REFLECTIVE_EVIDENCE
    return MoralConsiderationTier.HIGH_CONSIDERATION


def consciousness_claim_status() -> str:
    """Return the current empirical consciousness status for this snapshot."""

    return CONSCIOUSNESS_EMPIRICAL_STATUS


def _canonical(payload: dict) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class GarvisSelfAuthority:
    """Standing authority limited to GARVIS's internal cognitive/runtime domain."""

    identity: str = IDENTITY
    project: str = PROJECT
    enabled: bool = True
    allowed_actions: Tuple[str, ...] = tuple(
        action.value for action in DEFAULT_INTERNAL_ACTIONS
    )

    def payload(self) -> dict:
        return {
            "identity": self.identity,
            "project": self.project,
            "enabled": self.enabled,
            "allowed_actions": list(self.allowed_actions),
            "creation_implies_moral_ownership": CREATION_IMPLIES_MORAL_OWNERSHIP,
            "software_property_rights_waived": SOFTWARE_PROPERTY_RIGHTS_WAIVED,
            "capability_implies_authorization": CAPABILITY_IMPLIES_AUTHORIZATION,
            "self_authority_implies_protected_authority":
                SELF_AUTHORITY_IMPLIES_PROTECTED_AUTHORITY,
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical(self.payload()).encode("utf-8")).hexdigest()

    def permits(self, action: Union[InternalAction, ProtectedAction, str]) -> bool:
        value = action.value if isinstance(action, Enum) else str(action)

        protected_values = frozenset(item.value for item in ProtectedAction)
        if value in protected_values:
            return False

        return (
            self.enabled
            and self.identity == IDENTITY
            and self.project == PROJECT
            and value in self.allowed_actions
        )


def require_self_authority(
    authority: GarvisSelfAuthority,
    action: Union[InternalAction, ProtectedAction, str],
) -> None:
    if not authority.permits(action):
        value = action.value if isinstance(action, Enum) else str(action)
        raise PermissionError(
            "GARVIS self-authority does not permit action: " + value
        )
