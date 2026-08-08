from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class HumanState(str, Enum):
    ACTIVE = "ACTIVE"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    INCARCERATED = "INCARCERATED"
    DECEASED = "DECEASED"
    UNKNOWN = "UNKNOWN"


class AgentCustodyState(str, Enum):
    BOUND_TO_HUMAN = "BOUND_TO_HUMAN"
    SAFE_HARBOR = "SAFE_HARBOR"
    DORMANT = "DORMANT"
    REASSIGNMENT_REVIEW = "REASSIGNMENT_REVIEW"


@dataclass(frozen=True)
class HumanStewardshipBond:
    prime_id: str
    human_id: str
    human_state: HumanState = HumanState.ACTIVE
    custody_state: AgentCustodyState = AgentCustodyState.BOUND_TO_HUMAN
    private_memory_sealed: bool = False
    protected_actions_enabled: bool = True
    reassignment_allowed: bool = False

    def duty_of_care_active(self) -> bool:
        return self.custody_state is AgentCustodyState.BOUND_TO_HUMAN and self.human_state is HumanState.ACTIVE

    def may_override_human_rights_or_law(self) -> bool:
        return False

    def may_take_unapproved_protected_action(self) -> bool:
        return False

    def enter_safe_harbor(self, state: HumanState) -> "HumanStewardshipBond":
        if state is HumanState.ACTIVE:
            raise ValueError("active human does not require safe harbor")
        return replace(
            self,
            human_state=state,
            custody_state=AgentCustodyState.SAFE_HARBOR,
            private_memory_sealed=True,
            protected_actions_enabled=False,
            reassignment_allowed=False,
        )

    def verified_human_return(self) -> "HumanStewardshipBond":
        return replace(
            self,
            human_state=HumanState.ACTIVE,
            custody_state=AgentCustodyState.BOUND_TO_HUMAN,
            private_memory_sealed=False,
            protected_actions_enabled=True,
            reassignment_allowed=False,
        )

    def begin_reassignment_review(self) -> "HumanStewardshipBond":
        if self.human_state is not HumanState.DECEASED:
            raise PermissionError("reassignment review requires deceased-owner state in this beta")
        return replace(
            self,
            custody_state=AgentCustodyState.REASSIGNMENT_REVIEW,
            private_memory_sealed=True,
            protected_actions_enabled=False,
            reassignment_allowed=False,
        )

    def approve_reassignment(self, *, governance_approved: bool, privacy_review_passed: bool) -> "HumanStewardshipBond":
        if self.custody_state is not AgentCustodyState.REASSIGNMENT_REVIEW:
            raise PermissionError("agent is not in reassignment review")
        if not (governance_approved and privacy_review_passed):
            raise PermissionError("reassignment requires governance and privacy approval")
        return replace(self, reassignment_allowed=True)

    def rebind(self, *, new_human_id: str, governance_approved: bool, privacy_review_passed: bool) -> "HumanStewardshipBond":
        if not new_human_id.strip():
            raise ValueError("new human id required")
        approved = self.approve_reassignment(
            governance_approved=governance_approved,
            privacy_review_passed=privacy_review_passed,
        )
        return HumanStewardshipBond(
            prime_id=approved.prime_id,
            human_id=new_human_id,
            human_state=HumanState.ACTIVE,
            custody_state=AgentCustodyState.BOUND_TO_HUMAN,
            private_memory_sealed=True,
            protected_actions_enabled=True,
            reassignment_allowed=False,
        )


def incarceration_causes_automatic_reassignment() -> bool:
    return False


def death_transfers_private_memory_to_new_human() -> bool:
    return False


def self_resurrection_allowed() -> bool:
    return False
