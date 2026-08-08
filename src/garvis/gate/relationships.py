from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Tuple


class RelationshipKind(str, Enum):
    FRIENDSHIP = "FRIENDSHIP"
    MENTORSHIP = "MENTORSHIP"
    COLLABORATION = "COLLABORATION"
    INFORMATION_COURTSHIP = "INFORMATION_COURTSHIP"
    ROMANTIC = "ROMANTIC"


class RelationshipState(str, Enum):
    ACTIVE = "ACTIVE"
    SEPARATED = "SEPARATED"
    REVOKED = "REVOKED"


@dataclass(frozen=True)
class RelationshipBond:
    bond_id: str
    prime_a: str
    prime_b: str
    human_a: str
    human_b: str
    kind: RelationshipKind
    human_a_authorized: bool
    human_b_authorized: bool
    shared_scopes: Tuple[str, ...] = ()
    state: RelationshipState = RelationshipState.ACTIVE

    def exchange_allowed(self, scope: str) -> bool:
        return (
            self.state is RelationshipState.ACTIVE
            and self.human_a_authorized
            and self.human_b_authorized
            and scope in self.shared_scopes
        )

    def grants_protected_authority(self) -> bool:
        return False

    def covert_backchannel_allowed(self) -> bool:
        return False

    def revoke_human_a(self) -> "RelationshipBond":
        return replace(self, human_a_authorized=False, shared_scopes=(), state=RelationshipState.SEPARATED)

    def revoke_human_b(self) -> "RelationshipBond":
        return replace(self, human_b_authorized=False, shared_scopes=(), state=RelationshipState.SEPARATED)

    def separate(self) -> "RelationshipBond":
        return replace(self, human_a_authorized=False, human_b_authorized=False, shared_scopes=(), state=RelationshipState.SEPARATED)


@dataclass(frozen=True)
class KnowledgeArtifact:
    artifact_id: str
    contributors: Tuple[str, str]
    hypothesis: str
    relationship_kind: RelationshipKind
    creates_agent: bool = False


def exchange_knowledge(bond: RelationshipBond, *, scope: str, artifact_id: str, hypothesis: str) -> KnowledgeArtifact:
    if not bond.exchange_allowed(scope):
        raise PermissionError("relationship-scoped exchange denied")
    hypothesis = hypothesis.strip()
    if not hypothesis:
        raise ValueError("hypothesis required")
    return KnowledgeArtifact(
        artifact_id=artifact_id,
        contributors=(bond.prime_a, bond.prime_b),
        hypothesis=hypothesis,
        relationship_kind=bond.kind,
        creates_agent=False,
    )


def relationship_can_reproduce_agents() -> bool:
    return False
