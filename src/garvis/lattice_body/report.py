"""Shared records for one dry-run lattice cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Handoff:
    to: str
    text: str
    source_id: str = ""


@dataclass(frozen=True)
class Lesson:
    text: str
    evidence: str | None = None
    verified: bool = False
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    claim: str
    evidence_ptr: str | None
    expect_substring: str | None = None
    expect_sha256: str | None = None


@dataclass(frozen=True)
class LatticeCycleInput:
    task: str
    output_dir: Path
    workspace: Path
    agent: str = "GARVIS"
    handoffs: tuple[Handoff, ...] = ()
    lessons: tuple[Lesson, ...] = ()
    contradictions: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    provenance_report: Path | None = None
    law_zero_path: Path | None = None
    fib_target: int = 3
    prior_entry_count: int = 0
    act_tool: str = "file_read"
    act_path: str = ""
    memory: object | None = None


@dataclass
class CycleReport:
    steps: list[dict[str, str]]
    law_zero_ok: bool
    law_zero_text: str
    task: str
    output_dir: str
    creator: str
    body_proposes_only: bool = True
    heartbeat_consulted: bool = False
    operational_authorization: bool = False
    advisory_approved: bool = False
    merge_allowed: bool = False
    tools: list[dict[str, object]] = field(default_factory=list)
    memory_hints: list[dict[str, object]] = field(default_factory=list)
    memory_hint_b: float = 0.0
    echo_link: bool = False
    verifications: list[dict[str, object]] = field(default_factory=list)
    journal_items: list[dict[str, object]] = field(default_factory=list)
    scores: dict[str, object] = field(default_factory=dict)
    state: str = "FRAGMENTED"
    contradictions: list[str] = field(default_factory=list)
    open_handoffs: list[dict[str, str]] = field(default_factory=list)
    lessons: list[dict[str, object]] = field(default_factory=list)
    schedule_action: str = "wait"
    schedule_reason: str = ""
    fibonacci_adopted: bool = False
    closed: bool = False
    github_posting_enabled: bool = False
    github_posted: bool = False
    provenance_decision: str | None = None
    provenance_legal_conclusion: str | None = None
    halted: bool = False
    halt_reason: str = ""
    proposed_next_steps: list[dict[str, object]] = field(default_factory=list)
    approval_requests: list[str] = field(default_factory=list)
    digest_reason: str = "approval waiting"
    network_calls: int = 0
    withheld_contradicted: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "law_zero_ok": self.law_zero_ok,
            "task": self.task,
            "creator": self.creator,
            "body_proposes_only": self.body_proposes_only,
            "heartbeat_consulted": self.heartbeat_consulted,
            "operational_authorization": self.operational_authorization,
            "advisory_approved": self.advisory_approved,
            "advisory_approved_is_operational_authorization": False,
            "merge_allowed": self.merge_allowed,
            "tools": self.tools,
            "memory_hints": self.memory_hints,
            "memory_hint_b": self.memory_hint_b,
            "echo_prior_memory_link": self.echo_link,
            "verifications": self.verifications,
            "journal_items": self.journal_items,
            "scores": self.scores,
            "state": self.state,
            "contradictions": self.contradictions,
            "open_handoffs": self.open_handoffs,
            "lessons": self.lessons,
            "schedule_action": self.schedule_action,
            "schedule_reason": self.schedule_reason,
            "fibonacci_adopted": self.fibonacci_adopted,
            "closed": self.closed,
            "github_posting_enabled": self.github_posting_enabled,
            "github_posted": self.github_posted,
            "provenance_decision": self.provenance_decision,
            "provenance_legal_conclusion": self.provenance_legal_conclusion,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "proposed_next_steps": self.proposed_next_steps,
            "approval_requests": self.approval_requests,
            "digest_reason": self.digest_reason,
            "network_calls": self.network_calls,
            "withheld_contradicted": self.withheld_contradicted,
            "law_zero_text": self.law_zero_text,
        }
