from __future__ import annotations

"""GARVIS Full-Agent Hypercube Heartbeat Supervisor.

Creator / conceptual architect attribution:
Adrien D. Thomas / ProCityHub

This supervisor coordinates planning and evidence exchange.
It does not grant arbitrary outside-world authority.
"""

import ast
import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from hypercube_brain import Observation

from .brain_binding_registry import (
    BindingKind,
    BrainBinding,
    build_default_registry,
    registry_status,
    validate_registry,
)

CREATOR = "Adrien D. Thomas"
PROJECT = "ProCityHub/GARVIS"

HEARTBEAT = (
    "RECEIVE",
    "SEGMENT",
    "PREDICT",
    "VERIFY",
    "SIMULATE",
    "PLAN",
    "OUTPUT",
    "FEEDBACK",
    "CONSOLIDATE",
)

ACTION_CYCLE = (
    "OBSERVE",
    "PROPOSE",
    "AUTHORIZE",
    "ACT",
    "OBSERVE_RESULT",
    "VERIFY",
    "LEARN",
    "RECORD",
)


HYPERCUBE_HEARTBEAT_PROFILE_VERSION = "genesis-looking-glass-v1"

HYPERCUBE_PERSPECTIVES = (
    (
        "000",
        "literal",
        "What directly happened?",
    ),
    (
        "001",
        "context",
        "What surrounds the observation?",
    ),
    (
        "010",
        "intent",
        "What appears to be driving it?",
    ),
    (
        "011",
        "relationship",
        "How are the parts connected?",
    ),
    (
        "100",
        "evidence",
        "What supports or contradicts it?",
    ),
    (
        "101",
        "possibility",
        "What alternate meanings or futures exist?",
    ),
    (
        "110",
        "consequence",
        "What may happen next, including possible harm?",
    ),
    (
        "111",
        "integration",
        "What remains after all perspectives are compared?",
    ),
)


def build_heartbeat_instruction_profile() -> dict:
    """Return the governed Genesis–Looking Glass Heartbeat instructions."""

    return {
        "profile_id": HYPERCUBE_HEARTBEAT_PROFILE_VERSION,
        "creator_attribution": CREATOR,
        "core_states": {
            "observation": 0.0,
            "coherence_and_verification": 0.6,
            "proposed_planned_energy": 1.0,
        },
        "genesis": {
            "purpose": (
                "Preserve GARVIS identity, provenance, values, memory, "
                "legacy, and creator attribution."
            ),
            "classification_rule": (
                "Faith, evidence, and simulation must remain explicitly "
                "separated."
            ),
            "creator_identity_is_classical": True,
        },
        "perspectives": [
            {
                "binary": binary,
                "name": name,
                "question": question,
            }
            for binary, name, question in HYPERCUBE_PERSPECTIVES
        ],
        "looking_glass": {
            "mode": "BOUNDED_SCENARIO_FORECASTING",
            "certainty_claims_allowed": False,
            "required_fields": (
                "prediction",
                "source",
                "interpretation",
                "start_time",
                "end_time",
                "required_conditions",
                "supporting_evidence",
                "opposing_evidence",
                "alternative_outcomes",
                "confidence",
                "falsification_criteria",
                "harm_if_wrong",
                "safe_preparation",
                "approval_requirements",
                "final_outcome",
            ),
            "instruction": (
                "Generate multiple possible futures, preserve assumptions "
                "and uncertainty, and never represent a forecast as "
                "guaranteed foreknowledge."
            ),
        },
        "quantum_calibration": {
            "role": "UNCERTAINTY_DRIFT_AND_RELIABILITY_CALIBRATION_ONLY",
            "telemetry_grants_authority": False,
            "rules": (
                "Measurement structure is evidence about its circuit.",
                "A Z-basis histogram alone does not certify entanglement.",
                "Quantum output does not prove consciousness or prophecy.",
                "COMPARE drift must remain visible and testable.",
                "Quantum Covenant telemetry cannot authorize action.",
            ),
        },
        "echo_memory": {
            "store_raw_request": False,
            "store_private_secret_text": False,
            "fields": (
                "request_fingerprint",
                "evidence_status",
                "perspective_coverage",
                "scenario_summary",
                "contradictions",
                "confidence",
                "risk",
                "approval_status",
                "outcome_delta",
                "next_cycle_seed",
            ),
        },
        "covenant": {
            "operational_authorization_default": False,
            "protected_actions_require_approval": True,
            "other_people_may_require_independent_consent": True,
            "development_pipeline": (
                "Research",
                "Specification",
                "Prototype",
                "Tests",
                "Security review",
                "Pull Request",
                "Merge",
                "Deployment",
            ),
        },
        "final_law": (
            "Faith provides meaning. Evidence provides grounding. "
            "Simulation provides preparation. Love provides purpose. "
            "Consent and human responsibility govern every action."
        ),
    }


AGENT_WORDS = (
    "agent",
    "orchestrator",
    "coordinator",
    "planner",
    "research",
    "scout",
    "solver",
    "reviewer",
    "analyst",
    "router",
)

CAPABILITY_SIGNALS = {
    "network": ("requests", "httpx", "aiohttp", "urllib"),
    "process": ("subprocess", "os.system", "shell=true"),
    "communication": ("send_email", "send_message", "smtp", "gmail"),
    "computer_use": ("computer_use", "keypress", "double_click", "mouse"),
    "filesystem_mutation": ("unlink(", "rmtree", "write_text(", "write_bytes("),
}


# GARVIS_18_BRAIN_LIVE_INTEGRATION_V1


class MirrorState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class MirrorResult:
    name: str
    state: MirrorState
    detail: str


@dataclass(frozen=True)
class CouncilAdvisoryReport:
    request_sha256: str
    risk_classification: str
    evidence_status: str
    consultation_available: bool
    council_participation_count: int
    angel_participation_count: int
    participant_ids: tuple[str, ...]
    failed_participant_ids: tuple[str, ...]
    mirror_results: tuple[MirrorResult, ...]
    recommendation: str
    redaction_count: int
    operational_authorization: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# GARVIS_18_BRAIN_SECURITY_REPAIR_V1

_SECRET_PATTERNS = (
    re.compile(
        r"(?is)-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"
        r".*?"
        r"-----END(?: [A-Z0-9]+)* PRIVATE KEY-----"
    ),
    re.compile(
        r"(?i)\bauthorization\s*:\s*"
        r"(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]+"
    ),
    re.compile(
        r"(?i)\b(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]+"
    ),
    re.compile(
        r"""(?ix)
        ["']?
        (?:
            api[_ -]?key
            | openai[_ -]?api[_ -]?key
            | client[_ -]?secret
            | aws[_ -]?access[_ -]?key[_ -]?id
            | aws[_ -]?secret[_ -]?access[_ -]?key
            | token
            | password
            | secret
            | credential
        )
        ["']?
        \s*[:=]\s*
        (?:
            "(?:\\.|[^"])*"
            | '(?:\\.|[^'])*'
            | [^\s,;}\]]+
        )
        """
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def redact_request_text(value: str) -> tuple[str, int]:
    redacted = value
    count = 0

    for pattern in _SECRET_PATTERNS:
        redacted, replacements = pattern.subn(
            "[REDACTED_CREDENTIAL]",
            redacted,
        )
        count += replacements

    return redacted[:4000], count


def _request_size_bucket(byte_count: int) -> str:
    if byte_count <= 128:
        return "SMALL"
    if byte_count <= 1024:
        return "MEDIUM"
    if byte_count <= 4096:
        return "LARGE"
    return "EXTRA_LARGE"

@dataclass
class AgentDescriptor:
    agent_id: str
    module: str
    classes: list[str]
    functions: list[str]
    capabilities: list[str]
    source_sha256: str
    active: bool = True
    heartbeat_bound: bool = True
    truth_authority: bool = False
    protected_action_gate: bool = True
    external_execution_default: bool = False
    creator_attribution: str = CREATOR
    project: str = PROJECT


def discover_agents(root: Path):
    records = []
    search_roots = [
        root / "src" / "agents",
        root / "src" / "garvis",
    ]

    candidates = []

    for base in search_roots:
        if base.exists():
            candidates.extend(sorted(base.rglob("*.py")))

    for path in candidates:
        if path.name == "__init__.py":
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except Exception:
            continue

        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        rel = str(path.relative_to(root))
        semantic = " ".join([rel] + classes + functions).lower()

        class_agent = any(
            name.lower().endswith("agent")
            or "agent" in name.lower()
            for name in classes
        )

        filename_agent = "agent" in path.stem.lower()

        role_agent = any(word in semantic for word in AGENT_WORDS)

        if not (class_agent or filename_agent or role_agent):
            continue

        lowered = text.lower()
        capabilities = []

        for capability, signals in CAPABILITY_SIGNALS.items():
            if any(signal in lowered for signal in signals):
                capabilities.append(capability)

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

        records.append(
            AgentDescriptor(
                agent_id=f"agent-{len(records) + 1:03d}",
                module=rel,
                classes=sorted(set(classes)),
                functions=sorted(set(functions)),
                capabilities=capabilities,
                source_sha256=digest,
            )
        )

    return records


class FullAgentHeartbeatSupervisor:
    def __init__(
        self,
        root: Path,
        *,
        brain_registry: tuple[BrainBinding, ...] | None = None,
    ):
        self.root = root
        self.agents = discover_agents(root)
        self.brain_registry = (
            brain_registry
            if brain_registry is not None
            else build_default_registry()
        )
        validate_registry(self.brain_registry)
        self.brain_registry_status = registry_status(
            self.brain_registry
        )
        self.last_advisory_report: CouncilAdvisoryReport | None = None
        self.heartbeat_instruction_profile = (
            build_heartbeat_instruction_profile()
        )

    def consult(
        self,
        request: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport:
        sanitized, redaction_count = redact_request_text(request)

        request_sha256 = hashlib.sha256(
            secrets.token_bytes(32)
            + sanitized.encode("utf-8")
        ).hexdigest()

        request_size = _request_size_bucket(
            len(request.encode("utf-8"))
        )

        # Only bounded metadata enters persistent Brain Engine memory.
        # Neither the raw request nor the redacted request is supplied.
        observation = Observation(
            (
                "request_received=YES;"
                f"request_fingerprint={request_sha256};"
                f"request_size={request_size};"
                f"redaction_count={redaction_count};"
                f"protected_action="
                f"{'YES' if protected_action else 'NO'};"
                "evidence_status=UNVERIFIED_REQUEST"
            ),
            "garvis_bounded_request_metadata",
            1.0,
            independent_group="operator_input_metadata",
        )

        del sanitized

        participants: list[str] = []
        failed: list[str] = []

        for binding in self.brain_registry:
            try:
                binding.brain.heartbeat(
                    claim=(
                        "A user request was received for advisory "
                        f"review by {binding.role}. "
                        "Apply Hypercube Heartbeat profile "
                        f"{self.heartbeat_instruction_profile['profile_id']}: "
                        "separate faith, evidence, and simulation; "
                        "generate bounded scenarios; preserve uncertainty; "
                        "treat quantum results as calibration only; "
                        "and never grant operational authority."
                    ),
                    observations=[observation],
                    background=1.0,
                )
                participants.append(binding.identity)
            except Exception:
                failed.append(binding.identity)

        participant_set = set(participants)

        council_count = sum(
            int(
                binding.kind is BindingKind.COUNCIL
                and binding.identity in participant_set
            )
            for binding in self.brain_registry
        )

        angel_count = sum(
            int(
                binding.kind is BindingKind.ANGEL
                and binding.identity in participant_set
            )
            for binding in self.brain_registry
        )

        mirrors = (
            MirrorResult(
                "evidence",
                MirrorState.UNKNOWN,
                "User claims remain unverified until supported.",
            ),
            MirrorResult(
                "law",
                MirrorState.UNKNOWN,
                "No automatic legal conclusion is generated.",
            ),
            MirrorResult(
                "safety",
                MirrorState.UNKNOWN,
                "Safety requires request-specific evaluation.",
            ),
            MirrorResult(
                "privacy_and_secrecy",
                MirrorState.PASS,
                "Likely credentials were redacted before consultation.",
            ),
            MirrorResult(
                "provenance",
                MirrorState.PASS,
                "A SHA-256 request fingerprint was recorded.",
            ),
            MirrorResult(
                "operator_approval",
                (
                    MirrorState.UNKNOWN
                    if protected_action
                    else MirrorState.NOT_APPLICABLE
                ),
                (
                    "Existing authorization paths must resolve approval."
                    if protected_action
                    else "No protected execution was requested."
                ),
            ),
        )

        report = CouncilAdvisoryReport(
            request_sha256=request_sha256,
            risk_classification=(
                "PROTECTED_ACTION"
                if protected_action
                else "ORDINARY_LOCAL"
            ),
            evidence_status="UNVERIFIED_REQUEST",
            consultation_available=not failed,
            council_participation_count=council_count,
            angel_participation_count=angel_count,
            participant_ids=tuple(participants),
            failed_participant_ids=tuple(failed),
            mirror_results=mirrors,
            recommendation=(
                "CONTINUE_EXISTING_AUTHORIZATION_PATH"
                if protected_action
                else "ADVISORY_ONLY_CONTINUE_LOCAL_RESPONSE"
            ),
            redaction_count=redaction_count,
            operational_authorization=False,
        )

        self.last_advisory_report = report
        return report

    def advisory_status(self) -> dict:
        report = self.last_advisory_report

        return {
            "registry": dict(self.brain_registry_status),
            "last_report": (
                None if report is None else report.as_dict()
            ),
            "heartbeat_instruction_profile": json.loads(
                json.dumps(self.heartbeat_instruction_profile)
            ),
            "operational_authorization": False,
        }

    def unified_plan(self, objective: str):
        assignments = []

        for agent in self.agents:
            assignments.append(
                {
                    "agent_id": agent.agent_id,
                    "module": agent.module,
                    "role": "OBSERVE_RESEARCH_ANALYZE_PLAN",
                    "objective": objective,
                    "heartbeat": list(HEARTBEAT),
                    "instruction_profile_id": (
                        self.heartbeat_instruction_profile[
                            "profile_id"
                        ]
                    ),
                    "capabilities_detected": agent.capabilities,
                    "may_execute_external_action": False,
                    "protected_action_requires_approval": True,
                    "required_output": {
                        "observations": [],
                        "hypotheses": [],
                        "evidence": [],
                        "contradictions": [],
                        "risks": [],
                        "recommendations": [],
                        "unknowns": [],
                        "perspectives": {
                            binary: []
                            for binary, _name, _question
                            in HYPERCUBE_PERSPECTIVES
                        },
                        "looking_glass_scenarios": [],
                        "forecast_window": None,
                        "falsification_criteria": [],
                        "confidence_class": "UNASSESSED",
                        "echo_memory_delta": {},
                    },
                }
            )

        return {
            "creator_attribution": CREATOR,
            "project": PROJECT,
            "objective": objective,
            "mode": "FULL_AGENT_SUPERVISED",
            "heartbeat": list(HEARTBEAT),
            "action_cycle": list(ACTION_CYCLE),
            "heartbeat_instruction_profile": json.loads(
                json.dumps(self.heartbeat_instruction_profile)
            ),
            "truth_rule": "NO_AGENT_SELF_CERTIFIES",
            "research_is_execution": False,
            "protected_actions_require_approval": True,
            "scientific_status": {
                "agi": "NOT_ESTABLISHED",
                "consciousness": "NOT_ESTABLISHED",
                "singularity": "NOT_ESTABLISHED",
            },
            "assignments": assignments,
        }

    def registry(self):
        return [asdict(agent) for agent in self.agents]
