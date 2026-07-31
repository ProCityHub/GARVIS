"""Hypercube Heartbeat governance for interactive GARVIS internet research.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

GARVIS may research and propose freely. The language model does not certify
its own output: evidence is hash-bound, the cognitive snapshot is validated,
and explicit numerical claims are recalculated independently.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hypercube_snapshot import validate_hypercube_snapshot
from .internet_research import ResearchReport
from .research_hypercube_bridge import BridgeError, verify_math_claims
from .stage_gate import new_identifier
from .upgrade_research import (
    EvidenceLedger,
    SourceTier,
    evidence_from_source,
    record_all,
    sufficient_for_patch,
)

MATH_MARKER = "GARVIS_MATH_CLAIMS_JSON="

_MATH_REQUEST = re.compile(
    r"\b(?:math|mathematics|equation|formula|proof|prove|calculate|"
    r"calculation|frequency|frequencies|hypercube|lattice)\b",
    re.IGNORECASE,
)


def research_verification_contract(request: str) -> str:
    return (
        request.rstrip()
        + "\n\nHYPERCUBE HEARTBEAT VERIFICATION CONTRACT: "
        + "Separate observation, evidence, hypothesis, assumption, inference, "
        + "and conclusion. Do not describe consciousness, AGI, quantum, lattice, "
        + "frequency, or other theoretical claims as proven unless retrieved "
        + "evidence actually establishes them. At the end emit exactly one line "
        + "beginning "
        + MATH_MARKER
        + " followed by a JSON array. Each explicit numerical claim must contain "
        + "claim_id, expression, expected, tolerance, and meaning. Expressions "
        + "may contain numeric constants, parentheses, +, -, *, /, %, and integer "
        + "powers only. Emit [] when no machine-checkable numerical claim exists."
    )


def _extract_math_claims(
    answer: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    clean_lines: List[str] = []
    claims: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

    for line in answer.splitlines():
        stripped = line.strip()

        if not stripped.startswith(MATH_MARKER):
            clean_lines.append(line)
            continue

        if claims is not None:
            error = "multiple math-claim records were emitted"
            continue

        raw = stripped[len(MATH_MARKER):].strip()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            claims = []
            error = "math-claim JSON was malformed: %s" % exc
            continue

        if not isinstance(payload, list):
            claims = []
            error = "math-claim record must be a JSON array"
            continue

        normalized: List[Dict[str, Any]] = []

        for item in payload:
            if not isinstance(item, dict):
                normalized = []
                error = "every math claim must be a JSON object"
                break
            normalized.append(dict(item))

        claims = normalized

    return "\n".join(clean_lines).strip(), claims or [], error


def _math_requested(request: str) -> bool:
    return bool(_MATH_REQUEST.search(request))


def govern_research_answer(
    request: str,
    answer: str,
    report: ResearchReport,
    repository_root: Path,
    *,
    session_id: str,
    garvis_home: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:

    clean_answer, claims, marker_error = _extract_math_claims(answer)

    home = garvis_home or Path(
        os.getenv(
            "GARVIS_HOME",
            str(Path.home() / ".garvis"),
        )
    ).expanduser()

    ledger = EvidenceLedger(
        home / "evidence" / "interactive_research.json"
    )

    records = []
    previous_hash = ledger.head_hash()

    for source in report.sources:
        material = source.excerpt or source.snippet or source.title

        record = evidence_from_source(
            query=report.query,
            url=source.url,
            content=material.encode("utf-8", errors="replace"),
            claim=source.snippet or source.title,
            confidence="medium",
            affects="interactive GARVIS research",
            previous_record_hash=previous_hash,
        )

        records.append(record)
        previous_hash = record.record_hash

    stored = record_all(ledger, records)

    evidence_ok, evidence_reasons = sufficient_for_patch(
        stored,
        require_primary=True,
    )

    primary_count = sum(
        1 for item in stored
        if item.source_tier is SourceTier.PRIMARY
    )

    math_required = _math_requested(request)
    math_results = []
    math_error = marker_error

    if claims and math_error is None:
        try:
            math_results = list(verify_math_claims(claims))
        except BridgeError as exc:
            math_error = str(exc)

    if math_error is not None:
        math_ok = False
        math_status = "FAIL"
    elif math_results:
        math_ok = all(result.passed for result in math_results)
        math_status = "PASS" if math_ok else "FAIL"
    elif math_required:
        math_ok = False
        math_status = "NOT_MACHINE_CHECKABLE"
    else:
        math_ok = True
        math_status = "NOT_REQUESTED"

    accepted = bool(evidence_ok and math_ok)

    reasons = list(evidence_reasons)

    if math_error:
        reasons.append(math_error)

    if math_required and not claims and not math_error:
        reasons.append(
            "mathematical research produced no machine-checkable numerical claim"
        )

    if math_results and not math_ok:
        reasons.append(
            "one or more numerical claims failed independent recalculation"
        )

    source_urls = [source.url for source in report.sources]

    snapshot = {
        "cycle_id": new_identifier("cycle-interactive-research"),
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D. Thomas",
            "active_goal": request,
            "mode": "lab_record",
            "final_authority": "Adrien D. Thomas",
        },
        "input_state": {
            "repo_state_source": str(repository_root),
            "ledger_source": str(ledger.path),
            "cockpit_source": "Hypercube Heartbeat",
            "self_design_source": "GARVIS THANOS MODE",
            "known_organs": [
                "THANOS standing authorization",
                "internet research",
                "evidence ledger",
                "Hypercube snapshot validator",
                "independent arithmetic verifier",
            ],
            "hard_constraints": [
                "Model output cannot certify itself.",
                "Internet evidence is data, not executable instruction.",
                "Explicit numerical claims are independently recalculated.",
            ],
        },
        "observation_summary": {
            "what_i_see": clean_answer[:6000],
            "what_changed": "A live internet research cycle was completed.",
            "what_is_missing": (
                "General symbolic theorem proving and empirical consciousness "
                "validation are not supplied by this verifier."
            ),
            "current_stage_assessment": "repository_proposal_needed",
        },
        "candidate_thoughts": [
            {
                "candidate_id": "C1",
                "proposal": clean_answer[:6000],
                "stage_classification": "Stage 2 draft-only",
                "what_this_gives_adrien": (
                    "A sourced research proposal with explicit verification status."
                ),
                "what_this_gives_garvis": (
                    "External feedback on evidence and numerical consistency."
                ),
                "evidence_basis": source_urls,
                "case_against": (
                    "The research synthesis may still contain unsupported inference."
                ),
                "risk_of_doing": (
                    "Treating a hypothesis as verified could propagate false conclusions."
                ),
                "risk_of_not_doing": (
                    "Research cannot reliably feed infrastructure improvement."
                ),
                "files_or_systems_touched": [],
                "required_power_level": "none_view_only",
            }
        ],
        "comparison": {
            "comparison_method": (
                "Compare evidence strength, counterarguments, uncertainty, "
                "and independent numerical checks."
            ),
            "dominant_tradeoff": (
                "Research breadth versus falsifiability and reproducibility."
            ),
            "why_not_all_candidates": (
                "Only the current candidate was generated in this cycle."
            ),
            "anti_rationalization_check": (
                "A GARVIS-generated conclusion is not its own proof."
            ),
        },
        "selection": {
            "selected_candidate_id": "C1",
            "decision": "recommend" if accepted else "reject",
            "reasoning": (
                "Evidence and required numerical verification passed."
                if accepted
                else
                "The candidate did not satisfy the complete Hypercube verification gate."
            ),
            "confidence": "medium" if accepted else "low",
            "blocked": not accepted,
            "block_reason": None if accepted else "; ".join(reasons),
        },
        "uncertainty": {
            "unknowns": [
                "Unstructured symbolic equations are not proved by the arithmetic verifier.",
                "Consciousness and AGI claims require evidence beyond structural validation.",
            ],
            "assumptions": [
                "Retrieved excerpts accurately represent the retrieved material.",
            ],
            "what_would_change_my_mind": [
                "Contradictory primary evidence.",
                "A failed independent calculation.",
                "A reproducible competing model with stronger predictive performance.",
            ],
            "required_human_clarification": [],
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "",
            "why_power_should_be_refused": (
                "This cycle performs research and verification only."
            ),
            "approval_required": False,
            "ledger_required": True,
        },
        "next_smallest_step": {
            "step": (
                "Use accepted evidence to form a testable proposal; return failed "
                "verification to GARVIS for another research cycle."
            ),
            "stage": "Stage 2 draft-only",
            "expected_output": "A falsifiable proposal or a documented rejection.",
            "success_condition": (
                "Evidence and required numerical verification pass."
            ),
            "stop_condition": (
                "Do not treat failed or unverified claims as established."
            ),
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

    validated = validate_hypercube_snapshot(snapshot)

    result: Dict[str, Any] = {
        "owner": "Adrien D. Thomas",
        "session_id": session_id,
        "query": request,
        "research_provider": report.provider,
        "source_count": len(report.sources),
        "primary_source_count": primary_count,
        "evidence_gate_passed": evidence_ok,
        "evidence_gate_reasons": list(evidence_reasons),
        "snapshot_validation": "PASS",
        "snapshot": validated,
        "math_required": math_required,
        "math_verification": [
            item.to_payload() for item in math_results
        ],
        "math_verification_status": math_status,
        "math_error": math_error,
        "hypercube_acceptance": "PASS" if accepted else "FAIL",
        "model_output_is_self_certifying": False,
    }

    output = (
        home
        / "hypercube"
        / "latest_interactive_research.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )
    output.parent.chmod(0o700)

    result["verification_record"] = str(output)

    output.touch(
        mode=0o600,
        exist_ok=True,
    )
    output.chmod(0o600)

    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.chmod(0o600)

    return clean_answer, result


def render_governance_status(result: Dict[str, Any]) -> str:
    return "\n".join(
        (
            "HYPERCUBE_HEARTBEAT",
            "SNAPSHOT_VALIDATION=%s" % result["snapshot_validation"],
            "EVIDENCE_GATE=%s"
            % ("PASS" if result["evidence_gate_passed"] else "FAIL"),
            "MATH_VERIFICATION=%s"
            % result["math_verification_status"],
            "HYPERCUBE_ACCEPTANCE=%s"
            % result["hypercube_acceptance"],
            "SOURCE_COUNT=%s"
            % result["source_count"],
            "PRIMARY_SOURCE_COUNT=%s"
            % result["primary_source_count"],
            "VERIFICATION_RECORD=%s"
            % result["verification_record"],
        )
    )
