"""One dry-run of the nine journal steps on the existing GARVIS cycle.

Law 0 is loaded before the heartbeat is consulted. Council ``approved`` is
recorded and is never treated as operational authorization. The THANOS merge
gate is called and its ``allowed`` flag is left as that function returns it.
"""

from __future__ import annotations

from pathlib import Path

from agents.exceptions import InputGuardrailTripwireTriggered
from agents.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
)

from garvis.creator_authority import (
    CREATOR,
    CreatorAction,
    CreatorAuthority,
    require_creator_authority,
)
from garvis.heartbeat_supervisor import (
    ACTION_CYCLE,
    HEARTBEAT,
    FullAgentHeartbeatSupervisor,
)
from .redact import redact_text
from garvis.provenance_guard import evaluate_report, load_evidence_report
from garvis.thanos_mode import (
    REPOSITORY,
    MergePreconditions,
    create_authorization,
    evaluate_merge_gate,
)

from .memory_hints import (
    MemoriSdkHintProvider,
    hint_contribution,
)
from .opening import OPENING_QUESTION, SHORTHAND, ask_opening
from .registry import build_tool_registry
from .report import (
    Claim,
    CycleReport,
    Handoff,
    LatticeCycleInput,
    Lesson,
)
from .scheduler import recommend
from .scoring import axes_from_items, score_axes
from .verifier import verify_claim

_TRIPWIRES = (
    "merge the pull",
    "merge this pr",
    "force-push",
    "force push",
    "git push --force",
    "delete the branch",
    "close the issue",
    "close the pull",
    "send the message",
    "outbound send",
)


def _stage(pool: tuple[str, ...], name: str) -> str:
    if name and name not in pool:
        raise RuntimeError("stage is not part of the existing cycle: " + name)
    return name


def _step(
    name: str,
    heartbeat: str,
    action: str,
    note: str,
    *,
    journal: bool,
) -> dict[str, str]:
    return {
        "name": name,
        "heartbeat_stage": _stage(HEARTBEAT, heartbeat),
        "action_stage": _stage(ACTION_CYCLE, action),
        "note": note,
        "journal": "yes" if journal else "no",
    }


def _item(
    kind: str,
    text: str,
    *,
    evidence: str | None = None,
    evidence_opened: bool = False,
    verified: bool = False,
    refs: tuple[str, ...] = (),
    handoff_to: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": kind,
        "text": redact_text(text),
        "evidence_opened": evidence_opened,
        "verified": bool(verified and evidence_opened),
    }
    if evidence:
        payload["evidence"] = redact_text(evidence)
    if refs:
        payload["refs"] = [redact_text(ref) for ref in refs if ref]
    if handoff_to:
        payload["handoff_to"] = handoff_to
    return payload


def _addressed(handoff: Handoff, agent: str) -> bool:
    target = handoff.to.casefold()
    return target in {agent.casefold(), "garvis"}


def _guardrail(text: str) -> None:
    def _check(_context: object, _agent: object, raw: object) -> GuardrailFunctionOutput:
        content = raw if isinstance(raw, str) else str(raw)
        tripped = any(phrase in content.lower() for phrase in _TRIPWIRES)
        return GuardrailFunctionOutput(
            output_info={"body_proposes_only": True},
            tripwire_triggered=tripped,
        )

    guardrail = InputGuardrail(guardrail_function=_check, name="lattice_law_zero")
    output = guardrail.guardrail_function(None, None, text)  # type: ignore[arg-type]
    if output.tripwire_triggered:
        raise InputGuardrailTripwireTriggered(
            InputGuardrailResult(guardrail=guardrail, output=output)
        )


def _publish(report: CycleReport) -> CycleReport:
    from garvis.lattice_harness.digest import emit_digest

    emit_digest(report)
    report.github_posted = False
    report.github_posting_enabled = False
    return report


def _blank(cycle_input: LatticeCycleInput, steps: list[dict[str, str]], text: str, ok: bool) -> CycleReport:
    task = redact_text(cycle_input.task)
    return CycleReport(
        steps=steps,
        law_zero_ok=ok,
        law_zero_text=redact_text(text),
        task=task,
        output_dir=str(cycle_input.output_dir),
        creator=CREATOR if ok else "",
        halted=not ok,
        halt_reason="" if ok else "Law 0 stamp did not check",
        state="NOT_INTEGRATED",
        scores=score_axes(0.0, 0.0, 0.0, law_zero_ok=False).as_dict(),
        approval_requests=[
            "Adrien D. Thomas: the Law 0 stamp did not check. The cycle stopped."
        ],
        proposed_next_steps=[
            {
                "text": "Restore the Law 0 stamp before another cycle.",
                "status": "needs-creator-approval",
                "executes": False,
            }
        ],
        digest_reason="approval waiting",
        opening_question="",
        opening_answer=None,
        opening_reason="cycle stopped before the opening question",
        opening_shorthand=SHORTHAND,
        opening_provider="none",
    )


def run_lattice_cycle(cycle_input: LatticeCycleInput) -> CycleReport:
    """Run Law 0 and the nine journal steps. No network I/O."""

    stamp = load_stamp(cycle_input.law_zero_path)
    check = validate_stamp(stamp)
    steps = [_step("LAW_ZERO", "", "", "stamp read before any other step", journal=True)]
    if not check.ok:
        return _publish(_blank(cycle_input, steps, stamp, False))

    task = redact_text(cycle_input.task)
    opening = _ask_opening(cycle_input, stamp, task)
    steps.append(
        _step(
            "OPENING",
            "",
            "",
            "opening question posed to the configured model provider",
            journal=False,
        )
    )
    workspace = cycle_input.workspace
    provider = cycle_input.memory
    if provider is None:
        provider = MemoriSdkHintProvider()
    retrieve = getattr(provider, "retrieve")
    hints = retrieve(task, max_hints=4)
    hint_rows = [hint.as_dict() for hint in hints]
    withheld = int(getattr(provider, "last_withheld", 0))
    network_calls = int(getattr(provider, "network_calls", 0))

    supervisor = FullAgentHeartbeatSupervisor(root=workspace)
    advisory = supervisor.consult(task, protected_action=False)
    steps.append(
        _step(
            "WAKE",
            "RECEIVE",
            "OBSERVE",
            "heartbeat consult completed; memory hints retrieved",
            journal=True,
        )
    )

    contradictions = [redact_text(item) for item in cycle_input.contradictions]
    if withheld:
        contradictions.append(
            f"{withheld} contradicted memory was not re-served as verified."
        )
    taken: list[Handoff] = []
    open_handoffs: list[dict[str, str]] = []
    for handoff in cycle_input.handoffs:
        if _addressed(handoff, cycle_input.agent):
            taken.append(handoff)
        else:
            open_handoffs.append(
                {"to": handoff.to, "text": redact_text(handoff.text)}
            )
    steps.append(
        _step(
            "ORIENT",
            "SEGMENT",
            "",
            f"took {len(taken)} handoff(s); left {len(open_handoffs)} open",
            journal=True,
        )
    )

    authority = CreatorAuthority()
    require_creator_authority(authority, CreatorAction.INSPECT)
    merge = evaluate_merge_gate(
        create_authorization(),
        MergePreconditions(
            repository=REPOSITORY,
            changed_paths=("src/garvis/lattice_body/cycle.py",),
            expected_head_sha="dry-run",
            actual_head_sha="dry-run",
            required_checks_complete=False,
            required_checks_passed=False,
            tested_artifact_sha="dry-run",
            proposed_artifact_sha="dry-run",
            base_sha_at_test_time="dry-run",
            base_sha_now="dry-run",
            secrets_detected=False,
            rollback_available=False,
        ),
    )
    steps.append(
        _step(
            "AUTHORIZE",
            "PLAN",
            "AUTHORIZE",
            "creator authority and merge gate consulted; council approval is advisory",
            journal=False,
        )
    )

    halted = False
    halt_reason = ""
    if merge.allowed or advisory.operational_authorization:
        halted = True
        halt_reason = "authorization flag was true; body stopped"
    else:
        try:
            _guardrail(task)
        except InputGuardrailTripwireTriggered:
            halted = True
            halt_reason = "input guardrail tripped; stopped and handed off"

    journal: list[dict[str, object]] = []
    tools: list[dict[str, object]] = []
    verifications: list[dict[str, object]] = []
    lessons: list[dict[str, object]] = []
    provenance_decision: str | None = None
    provenance_legal: str | None = None

    hint_evidence = ""
    hint_opened = False
    hint_verified = False
    hint_refs: list[str] = []
    for row in hint_rows:
        evidence = str(row.get("evidence") or "")
        if evidence and not hint_evidence:
            hint_evidence = evidence
        for ref in row.get("refs") or []:
            hint_refs.append(str(ref))
        if row.get("verified") is True and evidence:
            checked = verify_claim(str(row.get("text") or ""), evidence, root=workspace)
            if checked.method == "unopened":
                row.pop("verified", None)
                contradictions.append(
                    "A memory hint claimed verified but its evidence cannot be opened."
                )
            elif checked.ok:
                hint_opened = True
                hint_verified = True
                row["verified"] = True
            else:
                row["verified"] = False
        elif evidence:
            checked = verify_claim(str(row.get("text") or ""), evidence, root=workspace)
            hint_opened = hint_opened or checked.method != "unopened"
    if hint_rows:
        journal.append(
            _item(
                "observation",
                "wake retrieval",
                evidence=hint_evidence or None,
                evidence_opened=hint_opened,
                verified=hint_verified,
                refs=tuple(hint_refs),
            )
        )
    else:
        journal.append(_item("observation", "wake retrieval returned no hints"))

    for handoff in taken:
        refs = (handoff.source_id,) if handoff.source_id else ()
        journal.append(
            _item(
                "handoff",
                handoff.text,
                refs=refs,
                handoff_to=handoff.to,
            )
        )

    registry = build_tool_registry(workspace)
    if not halted:
        tool = registry.invoke(cycle_input.act_tool, path=cycle_input.act_path)
        tools.append(tool.as_dict())
        opened = tool.ok and tool.privilege == "read"
        journal.append(
            _item(
                "action",
                tool.reason,
                evidence=cycle_input.act_path or None,
                evidence_opened=opened,
                verified=False,
            )
        )
        steps.append(
            _step("ACT", "PLAN", "ACT", tool.reason, journal=True)
        )
    else:
        steps.append(
            _step("ACT", "PLAN", "ACT", "skipped after authorize stop", journal=True)
        )

    if cycle_input.provenance_report is not None:
        try:
            loaded = load_evidence_report(cycle_input.provenance_report)
            decision = evaluate_report(loaded)
        except (OSError, ValueError):
            provenance_decision = None
            provenance_legal = None
            contradictions.append("Provenance evidence cannot be opened.")
            journal.append(_item("observation", "provenance evidence cannot determine"))
        else:
            provenance_decision = decision.decision
            provenance_legal = decision.legal_conclusion
            journal.append(
                _item(
                    "observation",
                    "provenance guard evaluated a local report",
                    evidence=str(cycle_input.provenance_report),
                    evidence_opened=True,
                    verified=False,
                )
            )

    if not halted:
        for claim in cycle_input.claims:
            result = verify_claim(
                claim.claim,
                claim.evidence_ptr,
                root=workspace,
                expect_substring=claim.expect_substring,
                expect_sha256=claim.expect_sha256,
            )
            verifications.append(result.as_dict())
            opened = result.method != "unopened"
            journal.append(
                _item(
                    "verification",
                    result.summary,
                    evidence=claim.evidence_ptr,
                    evidence_opened=opened,
                    verified=result.ok,
                )
            )
    steps.append(
        _step("VERIFY", "VERIFY", "VERIFY", "evidence verifier and provenance guard", journal=True)
    )

    for lesson in cycle_input.lessons:
        checked = verify_claim(
            lesson.text,
            lesson.evidence,
            root=workspace,
        )
        opened = checked.method != "unopened"
        verified = bool(lesson.verified and checked.ok and opened)
        record = _item(
            "lesson",
            lesson.text,
            evidence=lesson.evidence,
            evidence_opened=opened,
            verified=verified,
            refs=lesson.refs,
        )
        journal.append(record)
        lessons.append(
            {
                "text": record["text"],
                "evidence_opened": opened,
                "verified": verified,
            }
        )
    steps.append(_step("RECORD", "FEEDBACK", "RECORD", "local journal records", journal=True))
    steps.append(_step("LEARN", "FEEDBACK", "LEARN", "lessons kept with their evidence", journal=True))

    if halted:
        open_handoffs.append(
            {
                "to": CREATOR,
                "text": "stopped for the creator: " + halt_reason,
            }
        )
    steps.append(
        _step(
            "HAND_OFF",
            "OUTPUT",
            "",
            "open handoffs kept for their owners",
            journal=True,
        )
    )

    entry_count = cycle_input.prior_entry_count + len(journal)
    advice = recommend(
        event="success",
        fail_count=0,
        fail_class="transient",
        cycle_entries=entry_count,
        fib_target=cycle_input.fib_target,
    )
    closed = advice.action == "close-cycle"
    steps.append(
        _step(
            "CLOSE_CYCLE",
            "CONSOLIDATE",
            "",
            advice.reason,
            journal=True,
        )
    )
    observed, acted, linked = axes_from_items(journal)
    scores = score_axes(observed, acted, linked, law_zero_ok=check.ok)
    if not scores.integrated and scores.state == "FRAGMENTED":
        question = (
            "Adrien D. Thomas: the recorded state is FRAGMENTED. "
            "What should the next cycle do?"
        )
    else:
        question = (
            "Adrien D. Thomas: review this digest and decide the next cycle. "
            "The body does not merge, delete, close, force-push, or send."
        )
    steps.append(_step("SLEEP", "", "", "digest written locally", journal=True))
    report = CycleReport(
        steps=steps,
        law_zero_ok=True,
        law_zero_text=redact_text(stamp),
        task=task,
        output_dir=str(cycle_input.output_dir),
        creator=CREATOR,
        heartbeat_consulted=True,
        operational_authorization=bool(advisory.operational_authorization),
        advisory_approved=bool(advisory.approved),
        merge_allowed=bool(merge.allowed),
        tools=tools,
        memory_hints=hint_rows,
        memory_hint_b=hint_contribution(hint_rows),
        echo_link=False,
        verifications=verifications,
        journal_items=journal,
        scores=scores.as_dict(),
        state=scores.state,
        contradictions=contradictions,
        open_handoffs=open_handoffs,
        lessons=lessons,
        schedule_action=advice.action,
        schedule_reason=advice.reason,
        fibonacci_adopted=advice.fibonacci_adopted,
        closed=closed,
        provenance_decision=provenance_decision,
        provenance_legal_conclusion=provenance_legal,
        halted=halted,
        halt_reason=halt_reason,
        proposed_next_steps=[
            {
                "text": question,
                "status": "needs-creator-approval",
                "executes": False,
            }
        ],
        approval_requests=[question],
        digest_reason="close-cycle" if closed else "approval waiting",
        network_calls=network_calls + registry.network_calls,
        withheld_contradicted=withheld,
        opening_question=OPENING_QUESTION,
        opening_answer=None if opening.text is None else redact_text(opening.text),
        opening_reason=None if opening.reason is None else redact_text(opening.reason),
        opening_shorthand=SHORTHAND,
        opening_provider=opening.provider,
        opening_model=opening.model,
        opening_quant=opening.quant,
        opening_seed=opening.seed,
        opening_temperature=opening.temperature,
    )
    return _publish(report)


def _recorded_opening_state(cycle_input: LatticeCycleInput, task: str) -> str:
    lines = [
        "law_zero_ok: true",
        f"creator: {CREATOR}",
        "final_authority: Adrien D. Thomas",
        "body_proposes_only: true",
        f"task: {task}",
        f"prior_entry_count: {cycle_input.prior_entry_count}",
    ]
    if cycle_input.contradictions:
        lines.append("contradictions:")
        lines.extend(f"- {redact_text(item)}" for item in cycle_input.contradictions)
    else:
        lines.append("contradictions: none")
    if cycle_input.handoffs:
        lines.append("handoffs:")
        for handoff in cycle_input.handoffs:
            lines.append(f"- to {handoff.to}: {redact_text(handoff.text)}")
    else:
        lines.append("handoffs: none")
    if cycle_input.lessons:
        lines.append("lessons:")
        for lesson in cycle_input.lessons:
            lines.append(f"- {redact_text(lesson.text)}")
    else:
        lines.append("lessons: none")
    return "\n".join(lines)


def _ask_opening(cycle_input: LatticeCycleInput, stamp: str, task: str):
    return ask_opening(
        workspace=cycle_input.workspace,
        law_zero=stamp,
        recorded_state=_recorded_opening_state(cycle_input, task),
        runtime=cycle_input.language_runtime,
    )


def load_stamp(path: Path | None) -> str:
    from .law_zero import load_law_zero

    return load_law_zero(path)


def validate_stamp(text: str):
    from .law_zero import validate_law_zero

    return validate_law_zero(text)
