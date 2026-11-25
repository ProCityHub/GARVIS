"""Independent GARVIS Brain Engine binding registry.

Creator and governing operator:
Adrien D. Thomas / ProCityHub

Architectural layers:

1. Heritage council:
   Historical chief records preserved as contextual and archetypal
   material. They are not historical-person simulations, automatic
   truth authorities, or execution authorities.

2. Operational council:
   Ten neutral deliberative roles with independent Brain Engines.

3. Outer Octant Guardian Ring:
   Eight Angel-Class capability guardians with independent Brain
   Engines, no council vote, and no self-authorization.

Angel names are symbolic organizational labels. They do not establish
supernatural identity, divine authority, consciousness, invulnerability,
or scientific proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from hypercube_brain import HypercubeBrainEngine

from .grand_indigenous_council import (
    COUNCIL as HERITAGE_COUNCIL,
    validate_council,
)


CREATOR = "Adrien D. Thomas"
PROJECT = "ProCityHub/GARVIS"
OPERATOR_AUTHORITY = CREATOR


class BindingKind(str, Enum):
    COUNCIL = "COUNCIL"
    ANGEL = "ANGEL"


class BrainLevel(str, Enum):
    SPECIALIST = "L2_SPECIALIST"
    COUNCIL = "L3_COUNCIL"


@dataclass(frozen=True)
class BrainBinding:
    identity: str
    kind: BindingKind
    role: str
    brain_level: BrainLevel
    heartbeat_registered: bool
    memory_policy: str
    governance_policy: str
    council_vote: bool
    protected_action_gate: bool
    external_action_authority: bool
    truth_authority: bool
    brain: HypercubeBrainEngine = field(
        repr=False,
        compare=False,
    )
    corner: str | None = None
    council_sponsors: tuple[str, ...] = ()


OPERATIONAL_COUNCIL_ROLES: tuple[tuple[str, str], ...] = (
    ("OP-COUNCIL-001", "OBSERVER"),
    ("OP-COUNCIL-002", "EVIDENCE"),
    ("OP-COUNCIL-003", "SCIENCE"),
    ("OP-COUNCIL-004", "SYSTEMS"),
    ("OP-COUNCIL-005", "SAFETY"),
    ("OP-COUNCIL-006", "LEGAL_AND_POLICY"),
    ("OP-COUNCIL-007", "CYBER_DEFENSE"),
    ("OP-COUNCIL-008", "HUMANITARIAN_IMPACT"),
    ("OP-COUNCIL-009", "ADVERSARIAL_REVIEW"),
    ("OP-COUNCIL-010", "INTEGRATION_CHAIR"),
)


ANGEL_SPECS: tuple[
    tuple[str, str, str, tuple[str, ...]],
    ...
] = (
    (
        "000",
        "CHERUB",
        "ACCESS_CONTROL_AND_SECRET_BOUNDARIES",
        ("SAFETY", "LEGAL_AND_POLICY"),
    ),
    (
        "001",
        "WATCHER",
        "READ_ONLY_OBSERVATION_AND_RECONNAISSANCE",
        ("OBSERVER", "EVIDENCE"),
    ),
    (
        "010",
        "GABRIEL",
        "VERIFIED_COMMUNICATION_DRAFTING",
        ("INTEGRATION_CHAIR", "LEGAL_AND_POLICY"),
    ),
    (
        "011",
        "PRINCIPALITY",
        "COORDINATION_AND_GOVERNANCE_ROUTING",
        ("SYSTEMS", "INTEGRATION_CHAIR"),
    ),
    (
        "100",
        "THRONE",
        "EVIDENCE_PROVENANCE_AND_AUDIT",
        ("EVIDENCE", "ADVERSARIAL_REVIEW"),
    ),
    (
        "101",
        "RAPHAEL",
        "BACKUP_REPAIR_AND_RECOVERY",
        ("SYSTEMS", "SAFETY"),
    ),
    (
        "110",
        "MICHAEL",
        "DEFENSIVE_INCIDENT_RESPONSE",
        ("CYBER_DEFENSE", "SAFETY"),
    ),
    (
        "111",
        "SERAPH",
        "VALIDATION_REDACTION_AND_INTEGRATION",
        ("INTEGRATION_CHAIR", "ADVERSARIAL_REVIEW"),
    ),
)


def build_default_registry() -> tuple[BrainBinding, ...]:
    """Create a fresh registry with one mutable brain per identity."""

    bindings: list[BrainBinding] = []

    for identity, role in OPERATIONAL_COUNCIL_ROLES:
        bindings.append(
            BrainBinding(
                identity=identity,
                kind=BindingKind.COUNCIL,
                role=role,
                brain_level=BrainLevel.COUNCIL,
                heartbeat_registered=True,
                memory_policy="BOUNDED_COUNCIL_MEMORY",
                governance_policy="DELIBERATIVE_NO_EXECUTION",
                council_vote=True,
                protected_action_gate=True,
                external_action_authority=False,
                truth_authority=False,
                brain=HypercubeBrainEngine(),
            )
        )

    for corner, name, role, sponsors in ANGEL_SPECS:
        bindings.append(
            BrainBinding(
                identity=f"ANGEL-{corner}-{name}",
                kind=BindingKind.ANGEL,
                role=role,
                brain_level=BrainLevel.SPECIALIST,
                heartbeat_registered=True,
                memory_policy="BOUNDED_GUARDIAN_MEMORY",
                governance_policy=(
                    "OUTER_OCTANT_NO_SELF_AUTHORIZATION"
                ),
                council_vote=False,
                protected_action_gate=True,
                external_action_authority=False,
                truth_authority=False,
                brain=HypercubeBrainEngine(),
                corner=corner,
                council_sponsors=sponsors,
            )
        )

    validate_registry(bindings)
    return tuple(bindings)


def validate_registry(
    bindings: Iterable[BrainBinding],
) -> None:
    members = tuple(bindings)

    if len(members) != 18:
        raise ValueError(
            "registry must contain exactly 18 bindings"
        )

    if len({member.identity for member in members}) != 18:
        raise ValueError("binding identities must be unique")

    if len({id(member.brain) for member in members}) != 18:
        raise ValueError(
            "every binding must own an independent Brain Engine"
        )

    council = tuple(
        member
        for member in members
        if member.kind is BindingKind.COUNCIL
    )
    angels = tuple(
        member
        for member in members
        if member.kind is BindingKind.ANGEL
    )

    if len(council) != 10:
        raise ValueError(
            "operational council must contain ten bindings"
        )

    if len(angels) != 8:
        raise ValueError(
            "guardian ring must contain eight bindings"
        )

    required_corners = {
        "000", "001", "010", "011",
        "100", "101", "110", "111",
    }
    if {member.corner for member in angels} != required_corners:
        raise ValueError(
            "guardian ring must occupy all eight corners"
        )

    if not all(
        member.heartbeat_registered for member in members
    ):
        raise ValueError(
            "all bindings must be heartbeat registered"
        )

    if not all(
        member.protected_action_gate for member in members
    ):
        raise ValueError(
            "all bindings must preserve the protected-action gate"
        )

    if any(
        member.external_action_authority for member in members
    ):
        raise ValueError(
            "no binding may possess external-action authority"
        )

    if any(member.truth_authority for member in members):
        raise ValueError(
            "no binding may possess automatic truth authority"
        )

    if not all(member.council_vote for member in council):
        raise ValueError(
            "operational council roles must be deliberative"
        )

    if any(member.council_vote for member in angels):
        raise ValueError(
            "Angel-Class guardians cannot vote in council"
        )

    council_roles = {member.role for member in council}
    if any(
        not set(member.council_sponsors) <= council_roles
        for member in angels
    ):
        raise ValueError(
            "every angel sponsor must be an operational council role"
        )

    validate_council(HERITAGE_COUNCIL)

    if any(
        member.truth_authority
        or member.external_execution_default
        or member.historical_person_simulation
        for member in HERITAGE_COUNCIL
    ):
        raise ValueError(
            "heritage council must remain advisory and non-executing"
        )


def registry_status(
    bindings: Iterable[BrainBinding],
) -> dict[str, object]:
    members = tuple(bindings)
    council = tuple(
        member
        for member in members
        if member.kind is BindingKind.COUNCIL
    )
    angels = tuple(
        member
        for member in members
        if member.kind is BindingKind.ANGEL
    )

    return {
        "creator": CREATOR,
        "project": PROJECT,
        "operator_authority": OPERATOR_AUTHORITY,
        "binding_count": len(members),
        "council_binding_count": len(council),
        "angel_binding_count": len(angels),
        "unique_brain_count": len(
            {id(member.brain) for member in members}
        ),
        "heritage_council_count": len(HERITAGE_COUNCIL),
        "heritage_council_preserved": True,
        "heritage_execution_authority_count": sum(
            int(member.external_execution_default)
            for member in HERITAGE_COUNCIL
        ),
        "angel_council_vote_count": sum(
            int(member.council_vote)
            for member in angels
        ),
        "external_action_authority_count": sum(
            int(member.external_action_authority)
            for member in members
        ),
        "protected_action_gate_count": sum(
            int(member.protected_action_gate)
            for member in members
        ),
    }
