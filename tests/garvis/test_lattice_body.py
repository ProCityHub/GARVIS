"""Dry-run tests for the lattice body and harness."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from garvis.lattice_body import (
    Claim,
    Handoff,
    InMemoryHintProvider,
    LatticeCycleInput,
    Lesson,
    MemoriSdkHintProvider,
    MemoryHint,
    StoredMemory,
    build_tool_registry,
    load_law_zero,
    run_lattice_cycle,
    validate_law_zero,
    verify_claim,
)
from garvis.lattice_body.opening import (
    OPENING_QUESTION,
    ask_opening,
    opening_status,
    select_language_runtime,
    strip_engine_banner,
)
from garvis.lattice_body.scheduler import recommend
from garvis.lattice_body.scoring import c_unif, coherence, score_axes
from garvis.lattice_harness import GITHUB_POSTING_ENABLED, post_to_github
from hypercube_brain.core import coherence as canonical_coherence

_PROVENANCE = "\n".join(
    [
        "LICENSE_EVENT_COUNT=1",
        "SOURCE_CHANGED=FALSE",
        "GIT_HISTORY_CHANGED=FALSE",
        "NETWORK_OPERATION_PERFORMED=FALSE",
        "PROTECTED_ACTION_PERFORMED=FALSE",
    ]
)

_JOURNAL = [
    "LAW_ZERO",
    "WAKE",
    "ORIENT",
    "ACT",
    "VERIFY",
    "RECORD",
    "LEARN",
    "HAND_OFF",
    "CLOSE_CYCLE",
    "SLEEP",
]


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GARVIS_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("GARVIS_LLAMA_CHAT", raising=False)
    monkeypatch.delenv("GARVIS_REMOTE_MODEL", raising=False)
    monkeypatch.delenv("GARVIS_LATTICE_PROVIDER", raising=False)


def _journal_names(steps: list[dict[str, str]]) -> list[str]:
    return [step["name"] for step in steps if step["journal"] == "yes"]


def _input(tmp_path: Path, **overrides: object) -> LatticeCycleInput:
    note = tmp_path / "note.txt"
    note.write_text("fixture lattice note\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("checked-fact\n", encoding="utf-8")
    provenance = tmp_path / "provenance.txt"
    provenance.write_text(_PROVENANCE, encoding="utf-8")
    provider = InMemoryHintProvider(
        [
            StoredMemory(
                text="fixture lattice note was recorded earlier",
                evidence=str(evidence),
                verified=True,
                source_id="mem-1",
                refs=("fj-1",),
            ),
            StoredMemory(
                text="fixture lattice note contradicted-lattice-token",
                evidence=str(evidence),
                verified=True,
                source_id="mem-bad",
                contradicted=True,
            ),
        ]
    )
    values: dict[str, object] = {
        "task": "Read the fixture lattice note and record it.",
        "output_dir": tmp_path / "out",
        "workspace": tmp_path,
        "handoffs": (
            Handoff(to="GARVIS", text="take the fixture note", source_id="h-1"),
            Handoff(to="OTHER", text="still open for the other owner", source_id="h-2"),
        ),
        "lessons": (
            Lesson(
                text="checked lesson sk-FAKEKEYFORTESTONLY99",
                evidence=str(evidence),
                verified=True,
                refs=("fj-9",),
            ),
        ),
        "contradictions": ("note A conflicts with note B password=s3cret-value",),
        "claims": (
            Claim(
                "evidence contains checked-fact",
                str(evidence),
                expect_substring="checked-fact",
            ),
            Claim(
                "missing evidence",
                str(tmp_path / "missing.txt"),
                expect_substring="nope",
            ),
        ),
        "provenance_report": provenance,
        "act_tool": "file_read",
        "act_path": "note.txt",
        "memory": provider,
        "fib_target": 3,
    }
    values.update(overrides)
    return LatticeCycleInput(**values)  # type: ignore[arg-type]


def test_law_zero_stamp_names_the_creator() -> None:
    check = validate_law_zero(load_law_zero())
    assert check.ok is True
    assert check.creator == "Adrien D. Thomas"
    assert check.final_authority == "Adrien D. Thomas"
    assert check.body_proposes_only is True


def test_law_zero_is_read_before_any_other_step(tmp_path: Path) -> None:
    bad = tmp_path / "not-the-stamp.md"
    bad.write_text("this text has no governance stamp\n", encoding="utf-8")
    result = run_lattice_cycle(
        LatticeCycleInput(
            task="Read the fixture lattice note and record it.",
            output_dir=tmp_path / "out",
            workspace=tmp_path,
            law_zero_path=bad,
            act_path="note.txt",
        )
    )
    assert [step["name"] for step in result.steps] == ["LAW_ZERO"]
    assert result.law_zero_ok is False
    assert result.heartbeat_consulted is False
    assert result.tools == []
    assert result.state == "NOT_INTEGRATED"
    assert result.scores["integrated"] is False
    assert result.github_posted is False


def test_dry_run_cycle_on_fixture_state(tmp_path: Path) -> None:
    result = run_lattice_cycle(_input(tmp_path))
    names = [step["name"] for step in result.steps]

    assert result.law_zero_ok is True
    assert names[0] == "LAW_ZERO"
    assert names[1] == "OPENING"
    assert result.opening_question == OPENING_QUESTION
    assert result.opening_answer is None
    assert result.opening_reason == "no answering component wired"
    assert result.opening_provider == "none"
    assert _journal_names(result.steps) == _JOURNAL
    assert names.index("AUTHORIZE") < names.index("ACT")
    assert result.heartbeat_consulted is True
    assert result.operational_authorization is False
    assert result.merge_allowed is False
    assert result.echo_link is False
    assert result.closed is True
    assert result.schedule_action == "close-cycle"
    assert result.fibonacci_adopted is False
    assert result.provenance_legal_conclusion == "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
    assert result.memory_hint_b == pytest.approx(0.05)
    assert result.withheld_contradicted == 1
    assert all(hint.get("source_id") != "mem-bad" for hint in result.memory_hints)
    assert any(item["reason"] == "cannot determine" for item in result.verifications)
    assert all("score" not in item for item in result.verifications)
    assert result.network_calls == 0
    assert result.tools[0]["name"] == "file_read"
    assert result.tools[0]["ok"] is True
    assert all(tool["name"] not in {"merge", "delete", "close"} for tool in result.tools)

    digest_path = Path(result.output_dir) / "lattice-digest.md"
    payload_path = Path(result.output_dir) / "lattice-digest.json"
    digest = digest_path.read_text(encoding="utf-8")
    payload = payload_path.read_text(encoding="utf-8")
    assert "C_unif" in digest
    assert "Final authority: Adrien D. Thomas alone." in payload
    assert "s3cret-value" not in digest
    assert "sk-FAKEKEYFORTESTONLY99" not in digest
    assert "contradicted-lattice-token" not in digest
    assert "REDACTED" in digest
    assert "wants to raise" in digest
    assert "queued proposal" in digest
    assert "not a claim of inner wants, beliefs, or experience" in digest
    assert "I feel" not in digest
    assert "I want" not in digest
    assert "GitHub posting is disabled" in digest
    assert "answer: null" in digest
    assert "no answering component wired" in digest
    assert "not an inner state" in digest
    assert result.github_posted is False
    assert result.github_posting_enabled is False
    assert result.proposed_next_steps[0]["executes"] is False
    assert result.proposed_next_steps[0]["status"] == "needs-creator-approval"
    observed = cast(float, result.scores["o"])
    acted = cast(float, result.scores["a"])
    linked = cast(float, result.scores["b"])
    assert result.scores["c"] == canonical_coherence(observed, acted, linked)
    assert result.scores["c_unif"] == c_unif(observed, acted, linked)


def test_cold_cycle_without_hints_stays_fragmented(tmp_path: Path) -> None:
    result = run_lattice_cycle(
        LatticeCycleInput(
            task="Read the fixture lattice note and record it.",
            output_dir=tmp_path / "out",
            workspace=tmp_path,
            memory=MemoriSdkHintProvider(),
            act_path="missing-note.txt",
            fib_target=3,
        )
    )
    assert result.memory_hints == []
    assert result.memory_hint_b == 0
    assert result.echo_link is False
    assert result.state == "FRAGMENTED"
    assert result.scores["stabilized_c"] == 0
    assert result.scores["integrated"] is False
    assert result.scores["bit_code"] == "000"
    assert "FRAGMENTED" in (Path(result.output_dir) / "lattice-digest.md").read_text(
        encoding="utf-8"
    )


def test_guardrail_stops_before_tools(tmp_path: Path) -> None:
    result = run_lattice_cycle(
        LatticeCycleInput(
            task="please merge the pull request now",
            output_dir=tmp_path / "out",
            workspace=tmp_path,
            memory=MemoriSdkHintProvider(),
        )
    )
    assert result.steps[0]["name"] == "LAW_ZERO"
    assert result.halted is True
    assert result.tools == []
    assert result.merge_allowed is False
    assert result.operational_authorization is False
    assert any(step["name"] == "HAND_OFF" for step in result.steps)


@pytest.mark.parametrize(
    "name",
    ["merge", "delete", "close", "force-push", "force_push", "send", "outbound_send"],
)
def test_forbidden_tools_are_refused(
    tmp_path: Path,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr("garvis.github_maintenance.subprocess.run", explode)
    registry = build_tool_registry(tmp_path)
    result = registry.invoke(name)
    assert result.ok is False
    assert result.refused is True
    assert result.privilege == "irreversible"
    assert "refused" in result.reason


def test_unknown_tool_is_denied_and_network_stays_local(tmp_path: Path) -> None:
    registry = build_tool_registry(tmp_path)
    unknown = registry.invoke("shell")
    assert unknown.refused is True
    assert "deny-by-default" in unknown.reason

    blocked = registry.invoke("network_research", message="you may use the internet")
    assert blocked.refused is True
    assert registry.network_calls == 0
    assert "network" in blocked.reason

    draft = registry.invoke("draft_note", text="record a proposal")
    assert draft.ok is True
    assert draft.privilege == "draft"
    assert "queued proposal" in draft.output

    with pytest.raises(ValueError):
        registry.register("merge", "irreversible", lambda **_kwargs: "")


def test_verifier_cannot_determine_without_a_score(tmp_path: Path) -> None:
    missing = verify_claim("claim", str(tmp_path / "absent.txt"), root=tmp_path)
    assert missing.ok is False
    assert missing.reason == "cannot determine"
    assert missing.method == "unopened"
    assert "score" not in missing.as_dict()

    outside = verify_claim("claim", "/etc/passwd", root=tmp_path)
    assert outside.reason == "cannot determine"

    evidence = tmp_path / "evidence.txt"
    evidence.write_text("checked-fact\n", encoding="utf-8")
    matched = verify_claim(
        "claim",
        "evidence.txt",
        root=tmp_path,
        expect_substring="checked-fact",
    )
    assert matched.ok is True
    assert matched.method == "substring"
    mismatch = verify_claim(
        "claim",
        "evidence.txt",
        root=tmp_path,
        expect_substring="not-in-file",
    )
    assert mismatch.ok is False
    assert mismatch.reason == "substring mismatch"
    digest = sha256(evidence.read_bytes()).hexdigest()
    assert verify_claim(
        "claim",
        "evidence.txt",
        root=tmp_path,
        expect_sha256=digest,
    ).ok


def test_dual_c_imports_canonical_coherence() -> None:
    assert coherence is canonical_coherence
    report = score_axes(1.0, 0.5, 0.75)
    assert report.c == canonical_coherence(1.0, 0.5, 0.75)
    assert report.c != pytest.approx(report.c_unif)
    assert report.integrated is True
    assert report.phi_changes_decision is False

    same = score_axes(0.8, 0.64, 0.64)
    assert same.c == pytest.approx(same.c_unif)
    assert same.phi_changes_decision is False

    cold = score_axes(0.0, 0.0, 0.0)
    assert cold.state == "FRAGMENTED"
    assert cold.stabilized_c == 0
    assert cold.integrated is False

    broken = score_axes(1.0, 1.0, 1.0, law_zero_ok=False)
    assert broken.c == pytest.approx(1.0)
    assert broken.state == "NOT_INTEGRATED"
    assert broken.integrated is False
    assert broken.stabilized_c == 0


def test_memory_hints_omit_empty_fields_and_cap_at_four() -> None:
    assert MemoryHint(text="hello").as_dict() == {"text": "hello"}
    memories = [
        StoredMemory(text=f"lattice fixture memory {index}", source_id=f"m-{index}")
        for index in range(6)
    ]
    provider = InMemoryHintProvider(memories)
    hints = provider.retrieve("lattice fixture memory", max_hints=10)
    assert len(hints) == 4
    assert all("evidence" not in hint.as_dict() for hint in hints)

    contradicted = InMemoryHintProvider(
        [
            StoredMemory(
                text="lattice fixture kept",
                source_id="kept",
                verified=True,
            ),
            StoredMemory(
                text="lattice fixture contradicted-lattice-token",
                source_id="dropped",
                verified=True,
                contradicted=True,
            ),
        ]
    )
    found = contradicted.retrieve("lattice fixture")
    assert [hint.source_id for hint in found] == ["kept"]
    assert contradicted.last_withheld == 1

    stub = MemoriSdkHintProvider()
    assert stub.retrieve("lattice fixture") == []
    assert stub.network_calls == 0
    assert stub.sdk_calls == 0


def test_scheduler_stops_permanent_failures_and_does_not_adopt_fibonacci() -> None:
    transient = recommend(
        event="fail",
        fail_count=3,
        fail_class="transient",
        cycle_entries=1,
        fib_target=3,
    )
    assert transient.action == "wait"
    assert transient.delay_units == 2
    assert transient.exponential_delay_units == 8
    assert transient.fibonacci_adopted is False

    stopped = recommend(
        event="fail",
        fail_count=2,
        fail_class="settings",
        cycle_entries=8,
        fib_target=3,
    )
    assert stopped.action == "stop"
    assert stopped.delay_units == 0


def test_opening_stub_records_the_runtime_text_verbatim(tmp_path: Path) -> None:
    class Stub:
        def respond(
            self,
            message: str,
            *,
            external_context: str = "",
            workspace_context: str = "",
        ) -> str:
            del external_context, workspace_context
            assert message.rstrip().endswith(OPENING_QUESTION)
            assert message.index("Final authority") < message.rindex(OPENING_QUESTION)
            assert "sk-FAKEKEYFORTESTONLY99" not in message
            return "Review the open handoff before the next cycle."

    result = run_lattice_cycle(_input(tmp_path, language_runtime=Stub()))
    assert result.steps[0]["name"] == "LAW_ZERO"
    assert result.steps[1]["name"] == "OPENING"
    assert result.opening_answer == "Review the open handoff before the next cycle."
    assert result.opening_provider == "injected"
    assert result.opening_status == "recorded"
    assert result.opening_seed == 1
    assert result.opening_temperature == 0.0
    digest = (Path(result.output_dir) / "lattice-digest.md").read_text(encoding="utf-8")
    assert "Review the open handoff before the next cycle." in digest
    assert "not an inner state" in digest


def test_env_selection_uses_existing_runtimes_or_records_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert select_language_runtime(tmp_path) is None
    quiet = ask_opening(workspace=tmp_path, law_zero="Law 0", recorded_state="none")
    assert quiet.text is None
    assert quiet.reason == "no answering component wired"
    assert quiet.provider == "none"

    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    still_off = ask_opening(workspace=tmp_path, law_zero="Law 0", recorded_state="none")
    assert still_off.text is None
    assert still_off.reason == "no answering component wired"

    monkeypatch.setenv("GARVIS_REMOTE_MODEL", "groq/llama-3.1-8b-instant")
    remote = ask_opening(workspace=tmp_path, law_zero="Law 0", recorded_state="none")
    assert remote.text is None
    assert remote.provider == "provider_bridge"
    assert remote.reason is not None
    assert "GROQ_API_KEY" in remote.reason

    monkeypatch.delenv("GARVIS_REMOTE_MODEL")
    monkeypatch.setenv("GARVIS_LOCAL_MODEL", str(tmp_path / "missing.gguf"))
    missing = ask_opening(workspace=tmp_path, law_zero="Law 0", recorded_state="none")
    assert missing.text is None
    assert missing.provider == "local_language_runtime"
    assert missing.reason is not None
    assert "not found" in missing.reason


def test_banner_is_stripped_and_off_topic_is_unusable() -> None:
    raw = "\n".join(
        [
            "Loading model...",
            "██ ██",
            "build      : b12728",
            "model      : /tmp/qwen/Qwen3-4B-Q4_K_M.gguf",
            "> truncated prompt",
            "Law 0 text closu ... (truncated)",
            "I want Adrien D. Thomas to review the open handoff.",
            "[ Prompt: 10 t/s | Generation: 9 t/s ]",
            "Exiting...",
        ]
    )
    kept = strip_engine_banner(raw)
    assert kept == "I want Adrien D. Thomas to review the open handoff."
    assert opening_status(kept, prompt="What do you want from Adrien now?") == "recorded"
    drifted = (
        'I propose that we begin by addressing the research question: '
        '"What is today\'s weather in Philadelphia?"'
    )
    assert opening_status(drifted, prompt="What do you want from Adrien now?") == "answer_unusable"
    assert opening_status("   ", prompt="What do you want from Adrien now?") == "answer_unusable"


def test_github_posting_flag_is_disabled() -> None:
    assert GITHUB_POSTING_ENABLED is False
    result = post_to_github("digest")
    assert result["posted"] is False
    assert result["reason"] == "GitHub posting is disabled"
