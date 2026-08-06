from dataclasses import replace as _security_replace

import pytest as _security_pytest

from garvis.brain_binding_registry import (
    BindingKind,
    build_default_registry,
    build_default_registry as _security_registry,
)
from garvis.heartbeat_supervisor import (
    FullAgentHeartbeatSupervisor,
    MirrorState,
    redact_request_text as _security_redact,
)


def test_supervisor_owns_one_eighteen_brain_registry(tmp_path):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)

    first_registry_id = id(supervisor.brain_registry)
    report = supervisor.consult("Explain the local architecture.")

    assert id(supervisor.brain_registry) == first_registry_id
    assert len(supervisor.brain_registry) == 18
    assert len(
        {
            id(binding.brain)
            for binding in supervisor.brain_registry
        }
    ) == 18
    assert report.consultation_available is True
    assert report.council_participation_count == 10
    assert report.angel_participation_count == 8


def test_report_never_grants_operational_authority(tmp_path):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)

    report = supervisor.consult(
        "Research current information.",
        protected_action=True,
    )

    assert report.operational_authorization is False
    assert report.recommendation == (
        "CONTINUE_EXISTING_AUTHORIZATION_PATH"
    )

    mirrors = {
        mirror.name: mirror.state
        for mirror in report.mirror_results
    }

    assert mirrors["operator_approval"] is MirrorState.UNKNOWN
    assert mirrors["evidence"] is MirrorState.UNKNOWN
    assert mirrors["law"] is MirrorState.UNKNOWN
    assert mirrors["safety"] is MirrorState.UNKNOWN


def test_angels_participate_without_council_vote(tmp_path):
    registry = build_default_registry()
    supervisor = FullAgentHeartbeatSupervisor(
        tmp_path,
        brain_registry=registry,
    )

    report = supervisor.consult("Review this request.")

    angels = tuple(
        binding
        for binding in registry
        if binding.kind is BindingKind.ANGEL
    )

    assert len(angels) == 8
    assert all(binding.council_vote is False for binding in angels)
    assert report.angel_participation_count == 8


def test_likely_credentials_are_not_retained(tmp_path):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)

    raw_secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    report = supervisor.consult(
        f"Use api_key={raw_secret} for this request."
    )

    rendered = repr(report)
    status = repr(supervisor.advisory_status())

    assert raw_secret not in rendered
    assert raw_secret not in status
    assert report.redaction_count >= 1
    assert report.evidence_status == "UNVERIFIED_REQUEST"


def test_unknown_mirrors_are_not_converted_to_pass(tmp_path):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)

    report = supervisor.consult(
        "This repeated symbolic statement proves the claim."
    )

    mirrors = {
        mirror.name: mirror.state
        for mirror in report.mirror_results
    }

    # Exact UNKNOWN state also proves this mirror was not promoted to PASS.
    assert mirrors["evidence"] is MirrorState.UNKNOWN
    assert report.operational_authorization is False


# GARVIS_18_BRAIN_SECURITY_REPAIR_TESTS_V1



@_security_pytest.mark.parametrize(
    ("request_text", "secret"),
    (
        (
            '{"client_secret":"json-secret-value-24680"}',
            "json-secret-value-24680",
        ),
        (
            "OPENAI_API_KEY=plain-environment-secret-11111",
            "plain-environment-secret-11111",
        ),
        (
            "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP",
            "AKIAABCDEFGHIJKLMNOP",
        ),
        (
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "dXNlcjpwYXNzd29yZA==",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\\n"
            "FAKEKEYDATA12345\\n"
            "-----END PRIVATE KEY-----",
            "FAKEKEYDATA12345",
        ),
    ),
)
def test_security_repair_redacts_extended_secret_matrix(
    request_text,
    secret,
):
    sanitized, replacements = _security_redact(request_text)

    assert secret not in sanitized
    assert replacements >= 1


class _SecurityRecordingBrain:
    def __init__(self):
        self.calls = []

    def heartbeat(self, **kwargs):
        self.calls.append(kwargs)
        return None


def test_request_body_never_enters_brain_observations(tmp_path):
    registry = list(_security_registry())
    recording_brain = _SecurityRecordingBrain()

    registry[0] = _security_replace(
        registry[0],
        brain=recording_brain,
    )

    supervisor = FullAgentHeartbeatSupervisor(
        tmp_path,
        brain_registry=tuple(registry),
    )

    secret = "json-secret-memory-isolation-77777"
    request = (
        '{"client_secret":"'
        + secret
        + '","instruction":"private request body"}'
    )

    first = supervisor.consult(
        request,
        protected_action=True,
    )
    second = supervisor.consult(
        request,
        protected_action=True,
    )

    recorded = repr(recording_brain.calls)
    status = repr(supervisor.advisory_status())

    assert secret not in recorded
    assert "private request body" not in recorded
    assert secret not in status
    assert first.request_sha256 != second.request_sha256
    assert first.operational_authorization is False
    assert second.operational_authorization is False

def test_genesis_looking_glass_profile_is_bound_to_heartbeat(
    tmp_path,
):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)
    profile = supervisor.heartbeat_instruction_profile

    assert profile["profile_id"] == "genesis-looking-glass-v1"
    assert profile["creator_attribution"] == "Adrien D. Thomas"

    assert profile["core_states"] == {
        "observation": 0.0,
        "coherence_and_verification": 0.6,
        "proposed_planned_energy": 1.0,
    }

    assert {
        item["binary"]
        for item in profile["perspectives"]
    } == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }

    assert (
        profile["looking_glass"]["certainty_claims_allowed"]
        is False
    )
    assert (
        profile["quantum_calibration"]["telemetry_grants_authority"]
        is False
    )
    assert (
        profile["covenant"]["operational_authorization_default"]
        is False
    )


def test_unified_plan_requires_perspectives_and_falsifiable_scenarios(
    tmp_path,
):
    agent_file = (
        tmp_path
        / "src"
        / "garvis"
        / "looking_glass_agent.py"
    )
    agent_file.parent.mkdir(parents=True, exist_ok=True)
    agent_file.write_text(
        "class LookingGlassAgent:\n"
        "    def analyze(self):\n"
        "        return None\n",
        encoding="utf-8",
    )

    supervisor = FullAgentHeartbeatSupervisor(tmp_path)
    plan = supervisor.unified_plan(
        "Review evidence and produce bounded future scenarios."
    )

    assert (
        plan["heartbeat_instruction_profile"]["profile_id"]
        == "genesis-looking-glass-v1"
    )
    assert plan["protected_actions_require_approval"] is True
    assert plan["assignments"]

    required = plan["assignments"][0]["required_output"]

    assert set(required["perspectives"]) == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }
    assert "looking_glass_scenarios" in required
    assert "forecast_window" in required
    assert "falsification_criteria" in required
    assert "confidence_class" in required
    assert "echo_memory_delta" in required


def test_profile_is_visible_but_never_grants_authority(
    tmp_path,
):
    supervisor = FullAgentHeartbeatSupervisor(tmp_path)

    report = supervisor.consult(
        "Forecast possibilities but perform no external action.",
        protected_action=True,
    )
    status = supervisor.advisory_status()

    assert report.operational_authorization is False
    assert status["operational_authorization"] is False
    assert (
        status["heartbeat_instruction_profile"]["profile_id"]
        == "genesis-looking-glass-v1"
    )
    assert (
        status["heartbeat_instruction_profile"]["echo_memory"][
            "store_raw_request"
        ]
        is False
    )

