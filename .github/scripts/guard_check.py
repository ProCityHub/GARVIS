#!/usr/bin/env python3
"""GARVIS repository guard checks.

Owner and final authority: Adrien D. Thomas (ProCityHub/GARVIS).

Ported from ProCityHub/hypercubeheartbeat `.github/scripts/guard_check.py`,
which established this mechanism for the organization. Four blocks:

1. Edits to frozen evidence artifacts listed in FROZEN_FILES.txt.
2. Edits to governance source listed in GOVERNED_FILES.txt without an
   explicit audit exception.
3. Completion markers asserted inside documents rather than emitted by a run.
4. Misuse of reserved empirical verdict language.

The critical property, inherited from the original: both ledgers are read
from the BASE branch, never the working tree. A pull request therefore
cannot unfreeze a file by editing the ledger in the same pull request. This
is what keeps the guard outside the set of things it guards.

Vocabulary rule, unchanged across the organization:
    PASS / FAIL              -> software test and check outcomes
    SUPPORTED / NOT_SUPPORTED -> pre-registered empirical outcomes only
The two are never mixed.

Exit status 0 prints GUARD PASS. Any failure prints GUARD FAILED and exits 1.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

BASE_REF = os.environ.get("GITHUB_BASE_REF") or "main"
FROZEN_LEDGER = "FROZEN_FILES.txt"
GOVERNED_LEDGER = "GOVERNED_FILES.txt"
AUDIT_EXCEPTION = re.compile(r"AUDIT-EXCEPTION:\s*\S+")


# --------------------------------------------------------------------------
# Git plumbing
# --------------------------------------------------------------------------


def run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def diff_base_ref() -> str:
    base = f"origin/{BASE_REF}"
    run(["git", "fetch", "origin", BASE_REF, "--depth=100"], check=False)

    probe = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()

    local = subprocess.run(
        ["git", "merge-base", BASE_REF, "HEAD"],
        text=True,
        capture_output=True,
    )
    if local.returncode == 0 and local.stdout.strip():
        return local.stdout.strip()

    print(f"WARN: no merge base for {base}; using {base} directly")
    return base


def changed_files(base: str) -> list[str]:
    output = run(["git", "diff", "--name-only", base, "HEAD"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def added_lines_for(base: str, path: str) -> list[str]:
    diff = run(["git", "diff", "-U0", base, "HEAD", "--", path], check=False)
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


def commit_messages(base: str) -> str:
    return run(["git", "log", "--format=%B", f"{base}..HEAD"], check=False)


# --------------------------------------------------------------------------
# Ledgers
# --------------------------------------------------------------------------


def parse_ledger(text: str) -> set[str]:
    entries = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.add(stripped)
    return entries


def read_base_ledger(name: str) -> set[str]:
    """Read a ledger from the base branch, never the working tree."""

    for ref in (f"origin/{BASE_REF}", BASE_REF):
        result = subprocess.run(
            ["git", "show", f"{ref}:{name}"],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return parse_ledger(result.stdout)
    return set()


def matches_ledger(path: str, ledger: set[str]) -> bool:
    """Return True when ``path`` is covered by a ledger entry.

    Entries ending in ``/`` cover a whole directory tree.
    """

    normalized = path.replace("\\", "/")
    for entry in ledger:
        if entry.endswith("/"):
            if normalized.startswith(entry):
                return True
        elif normalized == entry:
            return True
    return False


# --------------------------------------------------------------------------
# Retracted formula (org-wide retraction R-003)
# --------------------------------------------------------------------------

SCALAR_PHI_PATTERNS = [
    re.compile(
        r"\b(?:score|s)\s*=\s*(?:phi|PHI|\u03c6)\s*[*\u00d7\u00b7]\s*"
        r"(?:o|observer)\s*[*\u00d7\u00b7]\s*(?:a|actor)\s*[*\u00d7\u00b7]\s*"
        r"(?:b|bridge|environment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:phi|PHI|\u03c6)\s*[*\u00d7\u00b7]\s*(?:o|observer)\s*[*\u00d7\u00b7]\s*"
        r"(?:a|actor)\s*[*\u00d7\u00b7]\s*(?:b|bridge|environment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:o|observer)\s*[*\u00d7\u00b7]\s*(?:a|actor)\s*[*\u00d7\u00b7]\s*"
        r"(?:b|bridge|environment)\s*[*\u00d7\u00b7]\s*(?:phi|PHI|\u03c6)\b",
        re.IGNORECASE,
    ),
]


# --------------------------------------------------------------------------
# Claim boundary
# --------------------------------------------------------------------------

EMPIRICAL_SUPPORT_WORDS = re.compile(r"\b(SUPPORTED|NOT_SUPPORTED)\b")

#: Markers that assert whole-system status. These are produced by a run or
#: they are not true. A document may not assert them.
RESERVED_COMPLETION_MARKERS = (
    "THANOS_MODE_IMPLEMENTATION",
    "STANDING_AUTHORITY",
    "PERSISTENCE",
    "REVOCATION",
    "INTERNET_RESEARCH_INTERFACE",
    "ISOLATED_SELF_MODIFICATION",
    "AUTONOMOUS_REPAIR_LOOP",
    "GITHUB_PROVIDER",
    "GITHUB_CI_REPAIR",
    "MERGE_WHEN_GREEN_LOGIC",
    "AUTONOMOUS_GITHUB_WORKFLOW",
    "AUTONOMOUS_MERGE_WHEN_GREEN",
    "HEALTH_CHECK",
    "RUNTIME_HEALTH_CHECK",
    "ROLLBACK",
    "HEARTBEAT",
    "AUDITOR_IMMUNE_SYSTEM",
    "AUDIT_INTEGRITY",
    "PYTHON_39_COMPATIBILITY",
    "FULL_TEST_SUITE",
    "SECURITY_REVIEW",
    "CAPABILITY_REGISTRY",
)

_ASSERTED_PASS = re.compile(
    r"^\s*(?:[-*>]\s*)?(" + "|".join(RESERVED_COMPLETION_MARKERS) + r")\s*=\s*PASS\b"
)

#: Documents that may quote markers while describing the format itself.
_MARKER_DOC_ALLOWLIST = (
    "docs/GARVIS_THANOS_MODE.md",
    "docs/GARVIS_AUTONOMOUS_UPGRADE_ARCHITECTURE.md",
    "RETRACTIONS.md",
    "GOVERNED_FILES.txt",
    "FROZEN_FILES.txt",
)

_CLAIM_TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def is_document(path: str) -> bool:
    return Path(path).suffix.lower() in _CLAIM_TEXT_SUFFIXES


def is_claim_language_guard_path(path: str) -> bool:
    """Return True for claim-controlled documents."""

    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()

    if normalized.startswith("claims/"):
        return True
    if name in {"claims.json", "preregistration.md", "outcomes.md"}:
        return True
    if normalized.startswith("docs/") and (
        "claim" in name or "outcome" in name or "result" in name or "success" in name
    ):
        return True
    return False


def is_text_path(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".py",
        ".md",
        ".txt",
        ".rst",
        ".yml",
        ".yaml",
        ".json",
        ".csv",
        ".toml",
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    base = diff_base_ref()
    changed = changed_files(base)

    frozen = read_base_ledger(FROZEN_LEDGER)
    governed = read_base_ledger(GOVERNED_LEDGER)
    messages = commit_messages(base)
    exception_claimed = bool(AUDIT_EXCEPTION.search(messages))

    failures: list[str] = []
    notices: list[str] = []

    # 1. Frozen evidence artifacts. No override.
    frozen_changes = sorted(p for p in changed if matches_ledger(p, frozen))
    if frozen_changes:
        failures.append(
            "Frozen evidence artifacts changed:\n"
            + "\n".join(f"  - {p}" for p in frozen_changes)
            + "\nThese are frozen on "
            + BASE_REF
            + ". Editing the ledger in this pull request does not unfreeze them."
        )

    # 2. Governance source. Overridable only by an explicit, typed exception.
    governed_changes = sorted(
        p for p in changed if matches_ledger(p, governed) and p not in frozen_changes
    )
    if governed_changes:
        listing = "\n".join(f"  - {p}" for p in governed_changes)
        if exception_claimed:
            notices.append(
                "Governance source changed under a declared audit exception:\n"
                + listing
                + "\nCode-owner review is still required by branch protection."
            )
        else:
            failures.append(
                "Governance source changed without an audit exception:\n"
                + listing
                + "\nThese files define how changes are validated, so a cycle "
                "cannot self-certify them.\nTo proceed, Adrien D. Thomas adds a "
                "commit message line:\n  AUDIT-EXCEPTION: <reason>\n"
                "This guard is the early signal; code-owner review on "
                + BASE_REF
                + " is the enforcement."
            )

    # 3 and 4. Line-level content checks.
    for path in changed:
        if not is_text_path(path) or not Path(path).exists():
            continue

        added = added_lines_for(base, path)

        for line in added:
            for pattern in SCALAR_PHI_PATTERNS:
                if pattern.search(line):
                    failures.append(
                        f"{path}: retracted scalar-phi formula pattern added: {line.strip()}"
                    )

        if is_document(path) and path not in _MARKER_DOC_ALLOWLIST:
            for line in added:
                match = _ASSERTED_PASS.match(line)
                if match:
                    failures.append(
                        f"{path}: completion marker {match.group(1)}=PASS asserted inside a "
                        f"document.\n  {line.strip()}\n  This marker is produced by a run "
                        "or it is not true. Cite the run, or record the honest "
                        "state (NOT_IMPLEMENTED / FAIL)."
                    )

        if is_claim_language_guard_path(path):
            for line in added:
                if EMPIRICAL_SUPPORT_WORDS.search(line):
                    failures.append(
                        f"{path}: reserved empirical verdict language added inside "
                        f"a claim-controlled document: {line.strip()}\n  SUPPORTED / "
                        "NOT_SUPPORTED belong to pre-registered empirical "
                        "outcomes only."
                    )

    for notice in notices:
        print("GUARD NOTICE")
        print()
        print(notice)
        print()

    if failures:
        print("GUARD FAILED")
        print()
        for failure in failures:
            print(failure)
            print()
        raise SystemExit(1)

    print("GUARD PASS")
    print(f"Checked {len(changed)} changed file(s) against {BASE_REF}.")
    print(f"Frozen entries: {len(frozen)}. Governed entries: {len(governed)}.")


if __name__ == "__main__":
    main()
