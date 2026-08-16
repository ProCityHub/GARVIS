"""
GARVIS Agent Communication Bus

Agents communicate through GitHub Issues with structured labels and
a shared JSON coordination file. This is the "town square" where agents
post findings, ask each other questions, and coordinate responses.

Protocol:
  - Each agent has an ID (e.g., 'ci-fix-bot', 'test-agent', 'code-quality')
  - Messages are GitHub Issues with label 'agent-bus' + 'agent:{target}'
  - Broadcasts use label 'agent-bus' + 'agent:all'
  - The coordination JSON (agent_bus.json) tracks message state
  - The Town Square workflow runs periodically to route messages

Message types:
  - FINDING    — An agent discovered something relevant to others
  - QUESTION   — An agent needs input from another agent
  - ALERT      — An agent detected a problem that needs immediate attention
  - RESPONSE   — A reply to a previous message
  - COORDINATE — Town Square dispatch instruction to an agent
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================
# Agent Registry — who exists and what they do
# ============================================================

AGENTS = {
    "ci-fix-bot": {
        "name": "CI Fix Bot",
        "role": "Auto-fixes known CI failure patterns",
        "watches": ["ci-failure", "type-error", "lint-error"],
        "triggers": ["workflow_run:failure"],
    },
    "test-agent": {
        "name": "Test Agent",
        "role": "Runs and validates the test suite",
        "watches": ["test-failure", "coverage"],
        "triggers": ["push", "pull_request"],
    },
    "falsification-agent": {
        "name": "Falsification Agent",
        "role": "Self-falsification loops for diagnostic operations",
        "watches": ["falsification", "diagnostic"],
        "triggers": ["schedule", "workflow_dispatch"],
    },
    "numerical-stability-agent": {
        "name": "Numerical Stability Agent",
        "role": "Checks for numerical instabilities (EPS, near-zero)",
        "watches": ["numerical", "eps", "precision"],
        "triggers": ["schedule", "push"],
    },
    "code-quality-agent": {
        "name": "Code Quality Agent",
        "role": "Lint, format, and structural code quality checks",
        "watches": ["lint", "formatting", "code-smell"],
        "triggers": ["push", "pull_request"],
    },
    "nightly-agent": {
        "name": "Nightly Self-Test Agent",
        "role": "Comprehensive nightly self-test of all systems",
        "watches": ["nightly", "regression"],
        "triggers": ["schedule:nightly"],
    },
    "release-agent": {
        "name": "Release Agent",
        "role": "Manages releases and version bumps",
        "watches": ["release", "version", "publish"],
        "triggers": ["workflow_dispatch", "push:tag"],
    },
    "greeting-agent": {
        "name": "Greeting Agent",
        "role": "Interfaces with repository visitors",
        "watches": ["question", "inquiry"],
        "triggers": ["issues", "discussion"],
    },
    "ci-doctor": {
        "name": "CI Doctor",
        "role": "Diagnoses CI failures and creates knowledge base entries",
        "watches": ["ci-failure", "diagnosis"],
        "triggers": ["workflow_run:failure"],
    },
}


# ============================================================
# Message types
# ============================================================

@dataclass
class AgentMessage:
    """A message between agents, carried via GitHub Issues."""
    from_agent: str
    to_agent: str  # or 'all' for broadcast
    message_type: str  # FINDING | QUESTION | ALERT | RESPONSE | COORDINATE
    subject: str
    body: str
    related_issue: int | None = None  # parent issue if this is a response
    fingerprint: str | None = None  # link to pattern registry fingerprint
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ============================================================
# Bus state — the shared coordination file
# ============================================================

def load_bus(repo_root: Path) -> dict[str, Any]:
    """Load the agent bus state from the repo."""
    bus_path = repo_root / "ci-fix-bot" / "agent_bus.json"
    if bus_path.exists():
        with open(bus_path) as f:
            return json.load(f)
    return {
        "messages": [],
        "agent_status": {},
        "last_town_square": None,
        "version": 1,
    }


def save_bus(repo_root: Path, bus: dict[str, Any]) -> Path:
    """Save the agent bus state to the repo."""
    bus_path = repo_root / "ci-fix-bot" / "agent_bus.json"
    bus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bus_path, "w") as f:
        json.dump(bus, f, indent=2)
    return bus_path


def post_to_bus(
    repo_root: Path,
    message: AgentMessage,
) -> dict[str, Any]:
    """Post a message to the local bus state (does not create GitHub issue)."""
    bus = load_bus(repo_root)
    msg_dict = {
        "from_agent": message.from_agent,
        "to_agent": message.to_agent,
        "message_type": message.message_type,
        "subject": message.subject,
        "body": message.body,
        "related_issue": message.related_issue,
        "fingerprint": message.fingerprint,
        "timestamp": message.timestamp,
        "delivered": False,
        "issue_number": None,
    }
    bus["messages"].append(msg_dict)
    save_bus(repo_root, bus)
    return msg_dict


def read_bus_messages(
    repo_root: Path,
    agent_id: str,
    unread_only: bool = True,
) -> list[dict[str, Any]]:
    """Read messages addressed to an agent (or broadcasts to 'all')."""
    bus = load_bus(repo_root)
    messages = []
    for msg in bus.get("messages", []):
        if msg["to_agent"] == agent_id or msg["to_agent"] == "all":
            if unread_only and msg.get("delivered"):
                continue
            messages.append(msg)
    return messages


def mark_delivered(
    repo_root: Path,
    message_index: int,
    issue_number: int,
) -> None:
    """Mark a bus message as delivered (issue created)."""
    bus = load_bus(repo_root)
    if message_index < len(bus["messages"]):
        bus["messages"][message_index]["delivered"] = True
        bus["messages"][message_index]["issue_number"] = issue_number
        save_bus(repo_root, bus)


def update_agent_status(
    repo_root: Path,
    agent_id: str,
    status: str,
    findings: list[str] | None = None,
) -> None:
    """Update an agent's status in the bus."""
    bus = load_bus(repo_root)
    if "agent_status" not in bus:
        bus["agent_status"] = {}
    bus["agent_status"][agent_id] = {
        "status": status,
        "findings": findings or [],
        "last_active": datetime.now(timezone.utc).isoformat(),
    }
    save_bus(repo_root, bus)


# ============================================================
# GitHub issue creation (called from GitHub Actions)
# ============================================================

ISSUE_BODY_TEMPLATE = """\
## Agent Message

| Field | Value |
|-------|-------|
| **From** | {from_agent} |
| **To** | {to_agent} |
| **Type** | {message_type} |
| **Timestamp** | {timestamp} |
{fingerprint_row}{related_issue_row}

### Subject

{subject}

### Message

{body}

---

_Agent-to-agent communication via the GARVIS Agent Bus._
_This issue was created by `{from_agent}` and addressed to `{to_agent}`._
"""


def create_agent_issue(
    message: AgentMessage,
    repo: str,
    token: str,
) -> int | None:
    """Create a GitHub issue for an agent message.

    Returns the issue number, or None if creation failed.
    """
    fingerprint_row = ""
    if message.fingerprint:
        fingerprint_row = f"| **Pattern** | `{message.fingerprint}` |\n"

    related_issue_row = ""
    if message.related_issue:
        related_issue_row = (
            f"| **Related to** | #{message.related_issue} |\n"
        )

    body = ISSUE_BODY_TEMPLATE.format(
        from_agent=message.from_agent,
        to_agent=message.to_agent,
        message_type=message.message_type,
        timestamp=message.timestamp,
        fingerprint_row=fingerprint_row,
        related_issue_row=related_issue_row,
        subject=message.subject,
        body=message.body,
    )

    labels = ["agent-bus", f"agent:{message.to_agent}"]
    if message.message_type == "ALERT":
        labels.append("agent-alert")
    elif message.message_type == "QUESTION":
        labels.append("agent-question")

    # Create issue via GitHub API
    cmd = [
        "curl", "-s", "-X", "POST",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github.v3+json",
        f"https://api.github.com/repos/{repo}/issues",
        "-d", json.dumps({
            "title": (
                f"[{message.message_type}] "
                f"{message.from_agent} → "
                f"{message.to_agent}: {message.subject}"
            ),
            "body": body,
            "labels": labels,
        }),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        response = json.loads(result.stdout)
        return response.get("number")
    except (json.JSONDecodeError, KeyError):
        return None


def close_agent_issue(
    issue_number: int,
    repo: str,
    token: str,
    resolution: str = "resolved",
) -> bool:
    """Close an agent message issue."""
    cmd = [
        "curl", "-s", "-X", "PATCH",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github.v3+json",
        f"https://api.github.com/repos/{repo}/issues/{issue_number}",
        "-d", json.dumps({
            "state": "closed",
            "state_reason": "completed" if resolution == "resolved" else "not_planned",
        }),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        response = json.loads(result.stdout)
        return response.get("state") == "closed"
    except (json.JSONDecodeError, KeyError):
        return False


# ============================================================
# Coordination — the Town Square
# ============================================================

def coordinate(repo_root: Path, repo: str, token: str) -> dict[str, Any]:
    """Town Square: read all pending messages and route them.

    This is called by the town-square.yml workflow on a schedule.
    It:
    1. Reads all undelivered messages from the bus
    2. Creates GitHub issues for each (if not already created)
    3. Checks for cross-agent correlations (multiple agents reporting same issue)
    4. Posts a coordination summary
    """
    bus = load_bus(repo_root)
    messages = bus.get("messages", [])
    undelivered = [m for m in messages if not m.get("delivered")]

    routed = 0
    issues_created = []

    for i, msg in enumerate(messages):
        if msg.get("delivered"):
            continue

        # Create GitHub issue for this message
        agent_msg = AgentMessage(
            from_agent=msg["from_agent"],
            to_agent=msg["to_agent"],
            message_type=msg["message_type"],
            subject=msg["subject"],
            body=msg["body"],
            related_issue=msg.get("related_issue"),
            fingerprint=msg.get("fingerprint"),
            timestamp=msg.get("timestamp", ""),
        )

        issue_num = create_agent_issue(agent_msg, repo, token)
        if issue_num:
            mark_delivered(repo_root, i, issue_num)
            issues_created.append(issue_num)
            routed += 1

    # Check for correlations — multiple agents reporting the same fingerprint
    fingerprint_groups: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        fp = msg.get("fingerprint")
        if fp:
            if fp not in fingerprint_groups:
                fingerprint_groups[fp] = []
            fingerprint_groups[fp].append(msg)

    correlations = []
    for fp, msgs in fingerprint_groups.items():
        unique_agents = {m["from_agent"] for m in msgs}
        if len(unique_agents) > 1:
            correlations.append({
                "fingerprint": fp,
                "agents": list(unique_agents),
                "count": len(msgs),
                "subjects": [m["subject"] for m in msgs],
            })

    # Post coordination summary if there are correlations
    if correlations:
        coord_msg = AgentMessage(
            from_agent="town-square",
            to_agent="all",
            message_type="COORDINATE",
            subject=f"{len(correlations)} cross-agent correlation(s) detected",
            body=json.dumps(correlations, indent=2),
        )
        post_to_bus(repo_root, coord_msg)

    # Update town square timestamp
    bus = load_bus(repo_root)
    bus["last_town_square"] = datetime.now(timezone.utc).isoformat()
    save_bus(repo_root, bus)

    return {
        "routed": routed,
        "issues_created": issues_created,
        "correlations": correlations,
        "total_messages": len(messages),
        "undelivered_remaining": len(undelivered) - routed,
    }


# ============================================================
# Agent self-report — agents call this after their run
# ============================================================

def report(
    agent_id: str,
    status: str,
    findings: list[str] | None = None,
    messages: list[AgentMessage] | None = None,
    repo_root: str = ".",
) -> None:
    """Agent self-report after completing work.

    Updates the agent's status and posts any outgoing messages.
    Called from the end of each agent's workflow.
    """
    repo_path = Path(repo_root)
    update_agent_status(repo_path, agent_id, status, findings)

    if messages:
        for msg in messages:
            post_to_bus(repo_path, msg)


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("GARVIS Agent Communication Bus")
        print()
        print("Commands:")
        print("  post    <from> <to> <type> <subject> <body> [--repo <path>]")
        print("  read    <agent_id> [--repo <path>] [--all]")
        print("  route   [--repo <path>] [--github-repo <owner/repo>]")
        print("  status  <agent_id> <status> [--repo <path>] [--finding <text>]")
        print("  report  <agent_id> <status> [--repo <path>]")
        print("  list    [--repo <path>]")
        print("  agents")
        sys.exit(1)

    command = sys.argv[1]

    if command == "agents":
        print(f"\nGARVIS Agent Registry — {len(AGENTS)} agents\n")
        for aid, info in AGENTS.items():
            print(f"  {aid:30s}  {info['role']}")
        return

    if command == "post":
        from_agent = sys.argv[2]
        to_agent = sys.argv[3]
        msg_type = sys.argv[4]
        subject = sys.argv[5]
        body = sys.argv[6] if len(sys.argv) > 6 else ""
        repo_root = Path(".")

        if "--repo" in sys.argv:
            idx = sys.argv.index("--repo")
            repo_root = Path(sys.argv[idx + 1])

        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=msg_type,
            subject=subject,
            body=body,
        )
        result = post_to_bus(repo_root, msg)
        print(json.dumps(result, indent=2))

    elif command == "read":
        agent_id = sys.argv[2]
        repo_root = Path(".")

        if "--repo" in sys.argv:
            idx = sys.argv.index("--repo")
            repo_root = Path(sys.argv[idx + 1])

        unread_only = "--all" not in sys.argv
        messages = read_bus_messages(repo_root, agent_id, unread_only)
        print(json.dumps(messages, indent=2))

    elif command == "route":
        repo_root = Path(".")
        if "--repo" in sys.argv:
            idx = sys.argv.index("--repo")
            repo_root = Path(sys.argv[idx + 1])

        repo = os.environ.get("GITHUB_REPOSITORY", "")
        token = os.environ.get("GITHUB_TOKEN", "")
        if "--github-repo" in sys.argv:
            idx = sys.argv.index("--github-repo")
            repo = sys.argv[idx + 1]

        result = coordinate(repo_root, repo, token)
        print(json.dumps(result, indent=2))

    elif command == "status":
        agent_id = sys.argv[2]
        status = sys.argv[3]
        repo_root = Path(".")
        findings = []

        if "--repo" in sys.argv:
            idx = sys.argv.index("--repo")
            repo_root = Path(sys.argv[idx + 1])
        if "--finding" in sys.argv:
            idx = sys.argv.index("--finding")
            findings.append(sys.argv[idx + 1])

        update_agent_status(repo_root, agent_id, status, findings)
        print(f"Updated {agent_id} status to {status}")

    elif command == "list":
        repo_root = Path(".")
        if "--repo" in sys.argv:
            idx = sys.argv.index("--repo")
            repo_root = Path(sys.argv[idx + 1])

        bus = load_bus(repo_root)
        messages = bus.get("messages", [])
        print(f"\nAgent Bus — {len(messages)} messages\n")
        for m in messages:
            delivered = "✓" if m.get("delivered") else "✗"
            print(
                f"  [{delivered}] {m['from_agent']:20s} → "
                f"{m['to_agent']:20s}  {m['message_type']:10s}  "
                f"{m['subject'][:50]}"
            )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
