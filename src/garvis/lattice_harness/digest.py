"""Local lattice digest.

The digest reports recorded state. "Wants to raise" means a queued proposal
for Adrien D. Thomas. GitHub posting is disabled and is not implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

from garvis.lattice_body.redact import redact_text
from garvis.lattice_body.report import CycleReport

GITHUB_POSTING_ENABLED = False

_FOOTER = (
    "This is a state report of recorded artifacts. It is not a claim of "
    "inner wants, beliefs, or experience.\n"
    "The phrase \"wants to raise\" means a queued proposal for "
    "Adrien D. Thomas. It does not name an inner want.\n"
    "GitHub posting is disabled. This digest was not posted."
)


def post_to_github(digest: str) -> dict[str, object]:
    """Do not contact GitHub. The flag stays off."""

    del digest
    if GITHUB_POSTING_ENABLED:
        raise RuntimeError("GitHub posting is not implemented")
    return {
        "posted": False,
        "reason": "GitHub posting is disabled",
    }


def _blank_number(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _answer_lines(answer: str | None) -> list[str]:
    if answer is None:
        return ["answer: null"]
    fence = "```"
    while fence in answer:
        fence += "`"
    return ["answer:", fence + "text", answer, fence]


def _fraction(report: CycleReport) -> tuple[int, int, float]:
    items = report.journal_items
    verified = sum(1 for item in items if item.get("verified") is True)
    total = len(items)
    fraction = 0.0 if total == 0 else verified / total
    return verified, total, fraction


def render_markdown(report: CycleReport) -> str:
    scores = report.scores
    verified, total, fraction = _fraction(report)
    lines = [
        "# GARVIS lattice digest",
        "",
        "Law 0 was read first. Creator and final authority: "
        f"{report.creator or 'unconfirmed'}. The body proposes only.",
        "",
        "## Opening",
        f"Question: {report.opening_question or 'not posed'}",
        f"Shorthand: {report.opening_shorthand}",
        *_answer_lines(report.opening_answer),
        f"reason: {report.opening_reason or ''}",
        f"provider: {report.opening_provider}",
        f"model: {report.opening_model or ''}",
        f"quant: {report.opening_quant or ''}",
        f"seed: {_blank_number(report.opening_seed)}",
        f"temperature: {_blank_number(report.opening_temperature)}",
        "",
        "## Lattice",
        f"- O: {scores.get('o', 0)}",
        f"- A: {scores.get('a', 0)}",
        f"- B: {scores.get('b', 0)}",
        f"- C: {scores.get('c', 0)}",
        f"- C_unif: {scores.get('c_unif', 0)}",
        f"- stabilized_C: {scores.get('stabilized_c', 0)}",
        f"- state: {report.state}",
        f"- bit_code: {scores.get('bit_code', '')}",
        f"- formula: {scores.get('formula', '')}",
        f"- control: {scores.get('control_formula', '')}",
        f"- phi_changes_decision: {str(scores.get('phi_changes_decision', False)).lower()}",
        f"- hypothesis: {scores.get('hypothesis', '')}",
        f"- memoryHints contribution: {report.memory_hint_b} (cap 0.20)",
        "- Echo prior-memory link: not stamped",
        "",
        "## Contradictions",
    ]
    if report.contradictions:
        lines.extend(f"- {item}" for item in report.contradictions)
    else:
        lines.append("- None recorded.")
    lines.append("")
    lines.append("## Open handoffs")
    if report.open_handoffs:
        lines.extend(
            f"- {item.get('to', '')}: {item.get('text', '')}"
            for item in report.open_handoffs
        )
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Lessons"])
    if report.lessons:
        for lesson in report.lessons:
            marker = "evidence opened" if lesson.get("evidence_opened") else "lacks evidence"
            lines.append(f"- ({marker}) {lesson.get('text', '')}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Coverage",
            (
                f"Verified fraction: {verified}/{total} = {fraction:.3f}. "
                "This is coverage of recorded items, not a feeling."
            ),
            "",
            "## Proposed next steps",
        ]
    )
    if report.proposed_next_steps:
        for step in report.proposed_next_steps:
            lines.append(
                f"- [{step.get('status')}] queued proposal: {step.get('text')}"
            )
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Approval requests"])
    if report.approval_requests:
        for index, request in enumerate(report.approval_requests, start=1):
            lines.append(f"{index}. {request}")
    else:
        lines.append("1. None recorded.")
    if report.state == "FRAGMENTED":
        lines.extend(
            [
                "",
                "The recorded state is FRAGMENTED. The next action is a question "
                "for Adrien D. Thomas: what should the next cycle do?",
            ]
        )
    lines.extend(
        [
            "",
            "## Close",
            f"- digest reason: {report.digest_reason}",
            f"- schedule: {report.schedule_action}",
            f"- fibonacci backoff adopted: {str(report.fibonacci_adopted).lower()}",
            f"- merge allowed: {str(report.merge_allowed).lower()}",
            "",
            "Wants to raise: the queued proposal in Approval requests.",
            "",
            _FOOTER,
            "",
        ]
    )
    return "\n".join(lines)


def emit_digest(report: CycleReport) -> dict[str, object]:
    """Write markdown, JSON, and a local journal. Do not post."""

    directory = Path(report.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    markdown = redact_text(render_markdown(report))
    payload = redact_text(
        json.dumps(report.as_dict(), sort_keys=True, indent=2, ensure_ascii=True)
    )
    journal_lines = [
        redact_text(json.dumps(item, sort_keys=True, ensure_ascii=True))
        for item in report.journal_items
    ]
    (directory / "lattice-digest.md").write_text(markdown, encoding="utf-8")
    (directory / "lattice-digest.json").write_text(payload + "\n", encoding="utf-8")
    (directory / "journal.jsonl").write_text(
        ("\n".join(journal_lines) + ("\n" if journal_lines else "")),
        encoding="utf-8",
    )
    posted = post_to_github(markdown)
    report.github_posted = False
    report.github_posting_enabled = False
    return {
        "markdown": str(directory / "lattice-digest.md"),
        "json": str(directory / "lattice-digest.json"),
        "journal": str(directory / "journal.jsonl"),
        "posted": posted["posted"],
    }
