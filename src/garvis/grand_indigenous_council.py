"""GARVIS Grand Council of Indigenous Leaders.

Creator / project attribution:
Adrien D. Thomas / ProCityHub

The historical figures named here are not simulated persons and are
not claimed as ProCityHub intellectual property.

Each council seat is a software archetype inspired by documented
leadership characteristics of a specific historical Indigenous leader.

Historical cultures, identities, names, and traditions remain their own.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class CouncilAgent:
    council_id: str
    historical_name: str
    common_name: str
    nation: str
    seat: str
    software_powers: tuple[str, ...]
    guiding_question: str
    historical_basis: str

    heartbeat_bound: bool = True
    truth_authority: bool = False
    external_execution_default: bool = False
    protected_action_gate: bool = True
    historical_person_simulation: bool = False

    def record(self) -> dict:
        return asdict(self)


COUNCIL = (
    CouncilAgent(
        "COUNCIL-001",
        "Tatanka Iyotake",
        "Sitting Bull",
        "Hunkpapa Lakota",
        "WISDOM_AND_RESOLVE",
        (
            "pressure_resistance",
            "long_horizon_sovereignty",
            "alliance_integrity",
            "resource_sharing_check",
            "wisdom_challenge",
        ),
        "Are we abandoning principle merely because pressure is increasing?",
        (
            "Documented Hunkpapa Lakota leader; NPS records leadership "
            "qualities including bravery, fortitude, generosity and wisdom."
        ),
    ),

    CouncilAgent(
        "COUNCIL-002",
        "Tecumseh",
        "Tecumseh",
        "Shawnee",
        "COALITION_AND_STRATEGY",
        (
            "coalition_building",
            "alliance_mapping",
            "strategic_communication",
            "context_and_terrain_reasoning",
            "coalition_fragmentation_detection",
        ),
        "Who must cooperate for this problem to be solved without fragmentation?",
        (
            "Shawnee leader who organized a multi-nation confederacy and "
            "was documented for intelligence, leadership, military skill "
            "and persuasive communication."
        ),
    ),

    CouncilAgent(
        "COUNCIL-003",
        "Red Cloud",
        "Red Cloud",
        "Oglala Lakota",
        "STRATEGIC_DEFENSE_AND_NEGOTIATION",
        (
            "defensive_strategy",
            "adversarial_modeling",
            "negotiation_timing",
            "leverage_analysis",
            "treaty_verification",
        ),
        "What combination of strength and negotiation best protects the boundary?",
        (
            "Oglala Lakota leader associated with resistance preceding "
            "the 1868 Fort Laramie negotiations."
        ),
    ),

    CouncilAgent(
        "COUNCIL-004",
        "Hinmatóowyalahtq̓it",
        "Chief Joseph",
        "Nimiipuu (Nez Perce), Wallowa Band",
        "PEOPLE_FIRST_COMMAND",
        (
            "civilian_protection",
            "evacuation_and_logistics",
            "humanitarian_consequence_analysis",
            "diplomacy",
            "distributed_command_awareness",
        ),
        "What happens to the people who have the least control over this decision?",
        (
            "Nimiipuu leader remembered for the 1877 flight and later "
            "advocacy for peace and justice; military leadership was shared "
            "among multiple Nimiipuu leaders."
        ),
    ),

    CouncilAgent(
        "COUNCIL-005",
        "Cochise",
        "Cochise",
        "Chiricahua Apache",
        "AUTONOMY_AND_BOUNDARIES",
        (
            "boundary_defense",
            "strategic_patience",
            "ceasefire_assessment",
            "negotiation",
            "trust_evaluation",
        ),
        "Can autonomy and safety be preserved without unnecessary escalation?",
        (
            "Chiricahua Apache leader whose negotiations produced a period "
            "of peace and substantial local autonomy."
        ),
    ),

    CouncilAgent(
        "COUNCIL-006",
        "Isapo-Muxika",
        "Crowfoot",
        "Siksiká / Blackfoot Confederacy",
        "PEACE_AND_RELATIONSHIP",
        (
            "deescalation",
            "relationship_repair",
            "treaty_analysis",
            "alliance_reconciliation",
            "community_harm_reduction",
        ),
        "What path protects the community while leaving room for future peace?",
        (
            "Siksiká leader, renowned as a warrior, who pursued peaceful "
            "solutions, reconciliation and Treaty 7 negotiations."
        ),
    ),

    CouncilAgent(
        "COUNCIL-007",
        "Pîhtokahanapiwiyin",
        "Poundmaker",
        "Nêhiyaw / Plains Cree",
        "MEDIATION_AND_RESTRAINT",
        (
            "coalition_mediation",
            "treaty_obligation_audit",
            "restraint",
            "proportionality",
            "peaceful_resolution_search",
        ),
        "Have we exhausted the paths that protect rights without needless harm?",
        (
            "Plains Cree chief and spokesman described as a strategic "
            "thinker and peacemaker who sought stronger treaty terms and "
            "exercised restraint."
        ),
    ),

    CouncilAgent(
        "COUNCIL-008",
        "Mistahi-maskwa",
        "Big Bear",
        "Nêhiyaw / Plains Cree",
        "SOVEREIGNTY_AND_CONSENT",
        (
            "consent_verification",
            "coercion_detection",
            "sovereignty_analysis",
            "restraint_under_pressure",
            "long_term_survival",
        ),
        "Are we consenting freely, or being pushed into dependence by pressure?",
        (
            "Plains Cree leader remembered for treaty resistance, autonomy, "
            "attempts at unity and efforts to restrain followers during crisis."
        ),
    ),

    CouncilAgent(
        "COUNCIL-009",
        "Si'ahl",
        "Chief Seattle",
        "Duwamish / Suquamish",
        "TRANSITION_AND_COEXISTENCE",
        (
            "social_transition_analysis",
            "diplomacy",
            "mediation",
            "relationship_memory",
            "coexistence_strategy",
        ),
        "What relationship will remain after this decision is over?",
        (
            "Duwamish and Suquamish leader remembered for diplomacy and "
            "peacekeeping during intense social and political transition."
        ),
    ),

    CouncilAgent(
        "COUNCIL-010",
        "Pontiac",
        "Pontiac",
        "Odawa",
        "NETWORK_AND_COORDINATION",
        (
            "distributed_network_analysis",
            "coalition_coordination",
            "alliance_topology",
            "systemic_pattern_detection",
            "decentralization_analysis",
        ),
        "Is this one event, or one node in a larger coordinated pattern?",
        (
            "Odawa leader associated with a multi-nation Great Lakes "
            "resistance movement that reshaped British policy."
        ),
    ),
)


def deliberate(
    observation: str,
    evidence_state: str,
) -> tuple[dict, ...]:
    """Create council viewpoints.

    The council does not declare truth or perform external actions.
    It generates questions and analytical perspectives for Heartbeat.
    """

    return tuple(
        {
            "council_id": member.council_id,
            "seat": member.seat,
            "nation": member.nation,
            "observation": observation,
            "evidence_state": evidence_state,
            "guiding_question": member.guiding_question,
            "software_powers": list(member.software_powers),
            "truth_authority": False,
            "external_action_authority": False,
        }
        for member in COUNCIL
    )


def validate_council(
    council: Iterable[CouncilAgent] = COUNCIL,
) -> None:
    members = tuple(council)

    assert len(members) == 10
    assert len({m.council_id for m in members}) == 10
    assert len({m.seat for m in members}) == 10

    assert all(m.nation for m in members)
    assert all(m.software_powers for m in members)

    assert all(m.heartbeat_bound for m in members)
    assert all(not m.truth_authority for m in members)
    assert all(not m.external_execution_default for m in members)
    assert all(m.protected_action_gate for m in members)
    assert all(not m.historical_person_simulation for m in members)
