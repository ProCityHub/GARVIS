"""Bounded autonomous repair engine for GARVIS THANOS MODE.

Creator / conceptual architect: Adrien D. Thomas.

GARVIS may research, propose and apply repository patches, validate them,
commit them and push a feature branch under an active THANOS authorization.
This module never merges or deploys.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from garvis.github_maintenance import GitHubMaintenanceAdapter
from garvis.thanos_mode import (
    ThanosAction,
    ThanosAuthorization,
    is_protected_path,
    permits,
)

DEFAULT_OBJECTIVE = (
    "Repair and improve GARVIS provider adapters, Universal AI Router, runtime "
    "integration, research/evidence plumbing, and Hypercube Heartbeat interfaces. "
    "Preserve evidence boundaries, existing behavior, tests, and governance."
)

_ALLOWED_PREFIXES = ("src/garvis/", "tests/garvis/")

_BLOCKED_REPAIR_PATHS = frozenset(
    {
        "src/garvis/autonomous_repair_runner.py",
        "src/garvis/thanos_cli.py",
        "src/garvis/thanos_mode.py",
        "src/garvis/upgrade_cycle.py",
        "src/garvis/stage_gate.py",
        "src/garvis/stage_gate_store.py",
        "src/garvis/github_maintenance.py",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _command(
    args: Sequence[str],
    repository: Path,
    *,
    timeout: int = 600,
    input_text: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=str(repository),
        text=True,
        input=input_text,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=dict(os.environ),
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def _branch(repository: Path) -> str:
    return _command(
        ["git", "branch", "--show-current"],
        repository,
        check=True,
    ).stdout.strip()


def _status(repository: Path) -> str:
    return _command(
        ["git", "status", "--porcelain"],
        repository,
        check=True,
    ).stdout


def _changed_files(repository: Path) -> list[str]:
    tracked = _command(
        ["git", "diff", "--name-only"],
        repository,
        check=True,
    )
    untracked = _command(
        ["git", "ls-files", "--others", "--exclude-standard"],
        repository,
        check=True,
    )
    return sorted(
        {
            line.strip()
            for output in (tracked.stdout, untracked.stdout)
            for line in output.splitlines()
            if line.strip()
        }
    )


def _configure_compatible_provider(model: str) -> None:
    if not model.casefold().startswith("compatible/"):
        return
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    if xai_key:
        os.environ.setdefault("GARVIS_COMPAT_API_KEY", xai_key)
        os.environ.setdefault("GARVIS_COMPAT_BASE_URL", "https://api.x.ai/v1")


def _candidate_models(requested: str | None = None) -> list[str]:
    """Return configured repair-model candidates in preference order."""
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        clean = value.strip()
        if clean and clean not in candidates:
            candidates.append(clean)

    add(requested)

    for value in os.getenv("GARVIS_REPAIR_MODELS", "").split(","):
        add(value)

    add(os.getenv("GARVIS_REPAIR_MODEL"))
    add(os.getenv("GARVIS_RESEARCH_MODEL"))
    add(os.getenv("GARVIS_MODEL"))

    if os.getenv("XAI_API_KEY", "").strip():
        add(os.getenv("GARVIS_XAI_MODEL") or "compatible/grok-4.5")

    if os.getenv("OPENAI_API_KEY", "").strip():
        add(os.getenv("GARVIS_OPENAI_MODEL") or "gpt-5.1")

    if os.getenv("OPENROUTER_API_KEY", "").strip():
        add(os.getenv("GARVIS_OPENROUTER_MODEL"))

    if os.getenv("GROQ_API_KEY", "").strip():
        add(os.getenv("GARVIS_GROQ_MODEL"))

    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        add(os.getenv("GARVIS_ANTHROPIC_MODEL"))

    return candidates


def _provider_label(model: str) -> str:
    lowered = model.casefold()

    if lowered.startswith(("anthropic/", "claude/", "claude-")):
        return "anthropic"
    if lowered.startswith("openrouter/"):
        return "openrouter"
    if lowered.startswith("groq/"):
        return "groq"
    if lowered.startswith(("grok/", "compatible/")):
        return "openai-compatible"
    if lowered.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
        return "openai"

    return "unknown"


def _context_bundle(repository: Path, limit: int = 180_000) -> str:
    paths: list[Path] = []

    explicit = (
        "src/garvis/provider_bridge.py",
        "src/garvis/universal_ai_registry.py",
        "src/garvis/capability_runtime.py",
        "src/garvis/assistant.py",
        "src/garvis/internet_research.py",
        "src/garvis/research_governance.py",
        "src/garvis/research_hypercube_bridge.py",
    )

    for name in explicit:
        path = repository / name
        if path.is_file():
            paths.append(path)

    patterns = (
        "tests/garvis/test_provider*.py",
        "tests/garvis/test_universal_ai_router*.py",
        "tests/garvis/test_research_hypercube*.py",
        "tests/garvis/test_internet_research*.py",
    )

    for pattern in patterns:
        paths.extend(sorted(repository.glob(pattern)))

    seen: set[str] = set()
    pieces: list[str] = []
    size = 0

    for path in paths:
        relative = path.relative_to(repository).as_posix()
        if relative in seen:
            continue
        seen.add(relative)

        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        block = f"\n===== {relative} =====\n{content}\n"
        if size + len(block) > limit:
            break
        pieces.append(block)
        size += len(block)

    return "".join(pieces)


def _research(
    repository: Path,
    model: str,
    objective: str,
) -> str:
    output = Path.home() / ".garvis" / "hypercube" / "autorepair_research.json"

    result = _command(
        [
            sys.executable,
            "-m",
            "garvis.research_hypercube_cli",
            "--model",
            model,
            "--repository",
            str(repository),
            "--output",
            str(output),
            objective,
        ],
        repository,
        timeout=300,
    )

    summary = (result.stdout + "\n" + result.stderr).strip()

    if result.returncode:
        raise RuntimeError(
            f"research provider {model!r} failed: "
            + summary[-6000:]
        )

    if output.is_file():
        try:
            evidence = output.read_text(encoding="utf-8")
            summary += "\n\nVERIFIED_RESEARCH_RECORD:\n" + evidence[:40_000]
        except OSError:
            pass

    return summary[-50_000:]


def _request_patch(
    repository: Path,
    model: str,
    objective: str,
    failures: str,
    research: str,
) -> str:
    """Ask one provider for a repair in an isolated Python process."""
    context = _context_bundle(repository, limit=90_000)

    prompt = f"""
You are GARVIS operating under Adrien D. Thomas's bounded autonomous
self-repair authority.

OBJECTIVE:
{objective}

HYPERCUBE HEARTBEAT:
RECEIVE -> SEGMENT -> PREDICT -> VERIFY -> SIMULATE -> PLAN ->
OUTPUT -> FEEDBACK -> CONSOLIDATE.

Use Observer / Actor / Background reasoning.
Evidence outranks theory.
External providers are candidate reasoning organs, never authority.

ONLINE RESEARCH / VERIFIED EVIDENCE:
{research}

CURRENT CHECK FAILURES:
{failures or "No previous failure. Inspect the adapter architecture and make one concrete useful repair."}

REPOSITORY CONTEXT:
{context}

Return ONE minimal unified git diff only.

Rules:
- May edit only src/garvis/** and tests/garvis/**.
- Do not edit THANOS, stage-gate, GitHub-maintenance, workflows,
  CODEOWNERS, governance, or the autonomous repair engine.
- Never include credentials or secret values.
- Preserve existing evidence and authorization boundaries.
- Do not merge or deploy.
- Do not weaken tests.
- Prefer a small repair to a rewrite.
- If no safe useful repair remains, output exactly: NO_PATCH_NEEDED
""".strip()

    worker = r"""
import asyncio
import sys
from pathlib import Path

from garvis.assistant import GarvisAssistant

model = sys.argv[1]
repository = Path(sys.argv[2]).resolve()
prompt = sys.stdin.read()

async def main():
    assistant = GarvisAssistant(
        model=model,
        persist_memory=False,
        repository_root=repository,
        max_turns=4,
    )
    reply = await assistant.respond(
        prompt,
        session_id="thanos-autorepair",
    )
    print(reply.text)

asyncio.run(main())
"""

    result = _command(
        [
            sys.executable,
            "-c",
            worker,
            model,
            str(repository),
        ],
        repository,
        input_text=prompt,
        timeout=420,
    )

    if result.returncode:
        raise RuntimeError(
            f"provider {model!r} failed:\n"
            + (result.stdout + "\n" + result.stderr)[-10000:]
        )

    response = result.stdout.strip()

    if not response:
        raise RuntimeError(
            f"provider {model!r} returned an empty response"
        )

    return response

def _extract_patch(response: str) -> str | None:
    if response.strip() == "NO_PATCH_NEEDED":
        return None

    fenced = re.search(
        r"```(?:diff|patch)?\s*\n(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate.startswith("diff --git "):
            return candidate + "\n"

    marker = response.find("diff --git ")
    if marker >= 0:
        return response[marker:].strip() + "\n"

    raise RuntimeError("GARVIS did not return a valid unified diff")


def _patch_paths(patch: str) -> list[str]:
    paths: list[str] = []

    for match in re.finditer(
        r"^diff --git a/(.+?) b/(.+?)$",
        patch,
        flags=re.MULTILINE,
    ):
        paths.append(match.group(2))

    if not paths:
        raise RuntimeError("patch contains no diff --git paths")

    return sorted(set(paths))


def _validate_patch(patch: str) -> list[str]:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(patch):
            raise RuntimeError("candidate patch appears to contain credential material")

    paths = _patch_paths(patch)

    for path in paths:
        if path in _BLOCKED_REPAIR_PATHS:
            raise RuntimeError(f"autonomous repair path is blocked: {path}")
        if is_protected_path(path):
            raise RuntimeError(f"governance-protected path is blocked: {path}")
        if not path.startswith(_ALLOWED_PREFIXES):
            raise RuntimeError(f"path outside autonomous repair scope: {path}")

    return paths


def _apply_patch(repository: Path, patch: str) -> list[str]:
    paths = _validate_patch(patch)

    checked = _command(
        ["git", "apply", "--check", "--whitespace=error-all", "-"],
        repository,
        input_text=patch,
    )
    if checked.returncode:
        raise RuntimeError(
            "candidate patch failed git apply --check:\n"
            + checked.stdout
            + checked.stderr
        )

    _command(
        ["git", "apply", "--whitespace=fix", "-"],
        repository,
        input_text=patch,
        check=True,
    )

    return paths


def _security_check(repository: Path, changed: Sequence[str]) -> str | None:
    for relative in changed:
        if relative in _BLOCKED_REPAIR_PATHS or is_protected_path(relative):
            return f"protected path changed: {relative}"

        path = repository / relative
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return f"possible credential material in {relative}"

    return None


def _checks(repository: Path) -> tuple[bool, str]:
    logs: list[str] = []
    failed = False

    commands: list[tuple[str, list[str]]] = [
        ("git-diff", ["git", "diff", "--check"]),
        (
            "compile",
            [sys.executable, "-m", "compileall", "-q", "src/garvis"],
        ),
    ]

    if importlib.util.find_spec("ruff") is not None:
        commands.append(
            (
                "ruff",
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "src/garvis",
                    "tests/garvis",
                ],
            )
        )

    focused = [
        path.as_posix()
        for pattern in (
            "tests/garvis/test_provider*.py",
            "tests/garvis/test_universal_ai_router*.py",
            "tests/garvis/test_research_hypercube*.py",
            "tests/garvis/test_internet_research*.py",
        )
        for path in sorted(repository.glob(pattern))
    ]

    if focused:
        commands.append(
            (
                "provider-hypercube-focused",
                [sys.executable, "-m", "pytest", "-q", *focused],
            )
        )

    commands.append(
        (
            "full-garvis",
            [sys.executable, "-m", "pytest", "-q", "tests/garvis"],
        )
    )

    for name, command in commands:
        result = _command(command, repository, timeout=1200)
        output = (result.stdout + "\n" + result.stderr).strip()
        logs.append(
            f"\n=== {name} rc={result.returncode} ===\n"
            + output[-12_000:]
        )
        if result.returncode:
            failed = True

    changed = _changed_files(repository)
    security_failure = _security_check(repository, changed)

    logs.append(
        "\n=== security ===\n"
        + ("PASS" if security_failure is None else f"FAIL: {security_failure}")
    )

    if security_failure:
        failed = True

    return not failed, "\n".join(logs)


def _write_report(payload: dict) -> Path:
    destination = (
        Path.home()
        / ".garvis"
        / "hypercube"
        / "autonomous_repair_latest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass
    return destination


def run_autonomous_repair(
    *,
    repository: Path,
    authorization: ThanosAuthorization,
    objective: str = DEFAULT_OBJECTIVE,
    model: str | None = None,
    max_repairs: int = 3,
) -> int:
    repo = repository.expanduser().resolve()

    permits(authorization, ThanosAction.CONTINUE_UPGRADING)
    permits(authorization, ThanosAction.INSPECT)
    permits(authorization, ThanosAction.RESEARCH)

    branch = _branch(repo)

    if branch.casefold() in {
        "main",
        "master",
        "production",
        "prod",
        "release",
    }:
        print(f"AUTONOMOUS_REPAIR=BLOCKED\nREASON=protected branch {branch}")
        return 2

    if _status(repo).strip():
        print("AUTONOMOUS_REPAIR=BLOCKED")
        print("REASON=worktree must be clean before GARVIS begins a new repair cycle")
        return 2

    if max_repairs < 1 or max_repairs > 10:
        print("AUTONOMOUS_REPAIR=BLOCKED")
        print("REASON=max_repairs must be between 1 and 10")
        return 2

    models = _candidate_models(model)

    if not models:
        print("AUTONOMOUS_REPAIR=BLOCKED")
        print("REASON=no configured repair-model candidates")
        return 2

    print("AUTONOMOUS_REPAIR=STARTED")
    print(f"BRANCH={branch}")
    print(
        "MODEL_CANDIDATES="
        + ",".join(
            f"{_provider_label(candidate)}:{candidate}"
            for candidate in models
        )
    )
    print(f"MAX_REPAIRS={max_repairs}")

    research = ""
    research_errors: list[str] = []

    for candidate in models:
        try:
            _configure_compatible_provider(candidate)
            research = _research(repo, candidate, objective)
            print(f"RESEARCH_MODEL={candidate}")
            break
        except Exception as exc:
            research_errors.append(f"{candidate}: {exc}")
            print(f"RESEARCH_PROVIDER_FAILED={candidate}: {exc}")

    if not research:
        research = (
            "ONLINE_RESEARCH_UNAVAILABLE\n"
            + "\n".join(research_errors)[-12000:]
        )

    failures = ""
    selected_model = models[0]

    for attempt in range(1, max_repairs + 1):
        print(f"HEARTBEAT_ATTEMPT={attempt}")
        print("PHASE=RECEIVE_SEGMENT_PREDICT_VERIFY_SIMULATE_PLAN")

        try:
            response = None
            provider_errors: list[str] = []

            for candidate in models:
                try:
                    print(
                        "REASONING_PROVIDER_TRY="
                        f"{_provider_label(candidate)}:{candidate}"
                    )

                    response = _request_patch(
                        repo,
                        candidate,
                        objective,
                        failures,
                        research,
                    )

                    selected_model = candidate
                    print(
                        f"REASONING_PROVIDER_SELECTED={candidate}"
                    )
                    break

                except Exception as provider_exc:
                    message = f"{candidate}: {provider_exc}"
                    provider_errors.append(message)
                    print(
                        f"REASONING_PROVIDER_FAILED={message}"
                    )

            if response is None:
                raise RuntimeError(
                    "all configured reasoning providers failed:\n"
                    + "\n".join(provider_errors)
                )

            patch = _extract_patch(response)

            if patch is None:
                report = _write_report(
                    {
                        "owner": "Adrien D. Thomas",
                        "status": "NO_PATCH_NEEDED",
                        "branch": branch,
                        "model": selected_model,
                        "attempt": attempt,
                        "objective": objective,
                    }
                )
                print("AUTONOMOUS_REPAIR=NO_PATCH_NEEDED")
                print(f"REPORT={report}")
                return 0

            paths = _apply_patch(repo, patch)
            print("PHASE=OUTPUT_FEEDBACK")
            print("PATCHED=" + ",".join(paths))

            passed, logs = _checks(repo)
            failures = logs

            if not passed:
                print("CHECKS=FAILED")
                print(logs[-20_000:])
                print("ACTION=REPAIR_AGAIN")
                continue

            changed = _changed_files(repo)
            if not changed:
                print("AUTONOMOUS_REPAIR=NO_CHANGES")
                return 0

            for path in changed:
                if path in _BLOCKED_REPAIR_PATHS or is_protected_path(path):
                    raise RuntimeError(f"protected path changed before commit: {path}")

            permits(authorization, ThanosAction.COMMIT)
            permits(authorization, ThanosAction.PUSH)

            adapter = GitHubMaintenanceAdapter(timeout_seconds=180)

            commit = adapter.commit_feature_changes(
                repo,
                branch,
                changed,
                f"fix(garvis): autonomous infrastructure repair {attempt}",
                approved=True,
            )

            push = adapter.push_feature_branch(
                repo,
                branch,
                approved=True,
            )

            report = _write_report(
                {
                    "owner": "Adrien D. Thomas",
                    "status": "PUSHED",
                    "branch": branch,
                    "model": selected_model,
                    "attempt": attempt,
                    "objective": objective,
                    "changed_files": changed,
                    "commit_output": commit.stdout.strip(),
                    "push_output": push.stdout.strip(),
                    "checks": logs,
                }
            )

            print("PHASE=CONSOLIDATE")
            print("CHECKS=PASS")
            print("COMMIT=PASS")
            print("PUSH=PASS")
            print(f"REPORT={report}")
            print("AUTONOMOUS_REPAIR=COMPLETED")
            return 0

        except Exception as exc:
            failures = str(exc)
            print(f"ATTEMPT_ERROR={exc}")

    report = _write_report(
        {
            "owner": "Adrien D. Thomas",
            "status": "BLOCKED",
            "branch": branch,
            "model": selected_model,
            "objective": objective,
            "last_failure": failures,
            "max_repairs": max_repairs,
        }
    )

    print("AUTONOMOUS_REPAIR=BLOCKED")
    print(f"REPORT={report}")
    return 1
