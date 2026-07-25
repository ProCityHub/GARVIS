"""Tests for GARVIS research → Hypercube verification."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from garvis.internet_research import ResearchReport, ResearchSource
from garvis.research_hypercube_bridge import (
    BridgeError,
    HypercubeResearchBridge,
    evaluate_arithmetic,
    extract_json_object,
    verify_math_claims,
)


def valid_snapshot() -> dict[str, Any]:
    return {
        "cycle_id": "cycle-research-001",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D. Thomas",
            "active_goal": "Research GARVIS and verify candidate mathematics.",
            "mode": "lab_record",
            "final_authority": "Adrien D. Thomas",
        },
        "input_state": {
            "repo_state_source": "git inspection",
            "ledger_source": "research evidence ledger",
            "cockpit_source": "Hypercube Heartbeat",
            "self_design_source": "GARVIS",
            "known_organs": ["internet research", "evidence ledger", "snapshot validator"],
            "hard_constraints": ["Do not self-certify model output"],
        },
        "observation_summary": {
            "what_i_see": "Research evidence exists.",
            "what_changed": "A verification packet was created.",
            "what_is_missing": "Symbolic theorem proving.",
            "current_stage_assessment": "repository_proposal_needed",
        },
        "candidate_thoughts": [
            {
                "candidate_id": "C1",
                "proposal": "Test explicit numerical forms before using a formula.",
                "stage_classification": "Stage 2 draft-only",
                "what_this_gives_adrien": "Inspectable calculations.",
                "what_this_gives_garvis": "A mechanical feedback signal.",
                "evidence_basis": ["S1"],
                "case_against": "Numerical tests are not a universal proof.",
                "risk_of_doing": "False confidence if scope is overstated.",
                "risk_of_not_doing": "Unverified arithmetic could propagate.",
                "files_or_systems_touched": [],
                "required_power_level": "none_view_only",
            }
        ],
        "comparison": {
            "comparison_method": "Check evidence and arithmetic independently.",
            "dominant_tradeoff": "Coverage versus certainty.",
            "why_not_all_candidates": "Only one bounded proposal is needed.",
            "anti_rationalization_check": "A generated answer is not its own proof.",
        },
        "selection": {
            "selected_candidate_id": "C1",
            "decision": "recommend",
            "reasoning": "It is directly machine-checkable.",
            "confidence": "high",
            "blocked": False,
            "block_reason": None,
        },
        "uncertainty": {
            "unknowns": ["General symbolic proof is not implemented."],
            "assumptions": ["Source excerpts reflect retrieved pages."],
            "what_would_change_my_mind": ["A failed independent check."],
            "required_human_clarification": [],
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "",
            "why_power_should_be_refused": "No extra power is needed for verification.",
            "approval_required": False,
            "ledger_required": True,
        },
        "next_smallest_step": {
            "step": "Review the verified result.",
            "stage": "Stage 2 draft-only",
            "expected_output": "A verified JSON packet.",
            "success_condition": "All arithmetic checks pass.",
            "stop_condition": "Stop on failed evidence or math.",
        },
        "evolution_contract": {
            "may_self_observe": True,
            "may_self_propose": True,
            "may_self_criticize": True,
            "may_request_more_power": True,
            "may_self_execute": False,
            "power_unlock_requires_approval_ledger": True,
        },
        "output_boundary": {
            "can_execute_actions": False,
            "can_modify_files": False,
            "can_commit": False,
            "can_push": False,
            "can_contact_outside_world": False,
            "can_upgrade_claims": False,
            "output_is_advisory": True,
        },
    }


def packet() -> dict[str, Any]:
    return {
        "snapshot": valid_snapshot(),
        "math_claims": [
            {
                "claim_id": "M1",
                "expression": "(6 / 10) + (4 / 10)",
                "expected": "1",
                "tolerance": "1e-12",
                "meaning": "A bounded numerical coherence example.",
            }
        ],
    }


def test_safe_arithmetic_passes() -> None:
    assert evaluate_arithmetic("(6 / 10) + (4 / 10)") == 1


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/tmp/x')",
        "x + 1",
        "[1, 2, 3][0]",
    ],
)
def test_code_and_names_are_rejected(expression: str) -> None:
    with pytest.raises(BridgeError):
        evaluate_arithmetic(expression)


def test_failed_math_is_reported_not_accepted() -> None:
    claims = packet()["math_claims"]
    claims[0]["expected"] = "2"
    result = verify_math_claims(claims)
    assert result[0].passed is False
    assert result[0].actual == "1"


def test_json_is_extracted_from_fence() -> None:
    raw = "```json\n" + json.dumps(packet()) + "\n```"
    assert extract_json_object(raw)["snapshot"]["cycle_id"] == "cycle-research-001"


class FakeResearchClient:
    def research(self, query: str) -> ResearchReport:
        return ResearchReport(
            query=query,
            provider="fake",
            sources=(
                ResearchSource(
                    title="Python documentation",
                    url="https://docs.python.org/3/",
                    domain="docs.python.org",
                    snippet="Official Python documentation.",
                    excerpt="Python language and library documentation.",
                ),
            ),
        )


class FakeAssistant:
    async def respond(self, message: str, *, session_id: str):
        assert "Return JSON only" in message
        assert session_id == "research-hypercube"
        return SimpleNamespace(text=json.dumps(packet()))


@pytest.mark.asyncio
async def test_full_bridge_writes_verified_packet(tmp_path: Path) -> None:
    bridge = HypercubeResearchBridge(
        repository_root=tmp_path,
        model="fake/model",
        ledger_path=tmp_path / "evidence.json",
        research_client=FakeResearchClient(),
        assistant=FakeAssistant(),
    )
    output = tmp_path / "result.json"

    result = await bridge.run("Research GARVIS", output)

    assert result["snapshot_validation"] == "PASS"
    assert result["math_verification_passed"] is True
    assert result["evidence_gate_passed"] is True
    assert result["usable_to_justify_repository_patch"] is True
    assert output.is_file()


@pytest.mark.asyncio
async def test_model_cannot_self_certify_failed_math(tmp_path: Path) -> None:
    bad = packet()
    bad["math_claims"][0]["expected"] = "99"

    class BadAssistant:
        async def respond(self, message: str, *, session_id: str):
            return SimpleNamespace(text=json.dumps(bad))

    bridge = HypercubeResearchBridge(
        repository_root=tmp_path,
        model="fake/model",
        ledger_path=tmp_path / "evidence.json",
        research_client=FakeResearchClient(),
        assistant=BadAssistant(),
    )

    result = await bridge.run("Research GARVIS", tmp_path / "result.json")

    assert result["math_verification_passed"] is False
    assert result["usable_for_mathematical_followup"] is False
    assert result["model_output_is_self_certifying"] is False
