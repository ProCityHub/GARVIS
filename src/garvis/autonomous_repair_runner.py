"""Bounded autonomous repair engine for GARVIS THANOS MODE.

Creator / conceptual architect: Adrien D. Thomas.

GARVIS may research, propose and apply repository patches, validate them,
commit them and push a feature branch under an active THANOS authorization.
This module never merges or deploys.
"""

from __future__ import annotations

import difflib
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
from garvis.hypercube_heartbeat import (
    AdaptiveWatchdog,
    HeartbeatState,
    PulseMetrics,
    ReleaseGate,
    detect_event_boundary,
)
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


def _local_runtime_available() -> bool:
    """Return True only when the local llama engine and GGUF brain are usable."""
    try:
        from garvis.local_language_runtime import LocalRuntimeConfig

        config = LocalRuntimeConfig.from_environment(Path.cwd())
        return (
            config.engine.is_file()
            and os.access(config.engine, os.X_OK)
            and config.model.is_file()
        )
    except Exception:
        return False


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

    if _local_runtime_available():
        add("local")

    return candidates


def _provider_label(model: str) -> str:
    lowered = model.casefold()

    if lowered in {"local", "garvis-local", "garvis/local"}:
        return "local"
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

        remaining = max(0, int(limit) - size)
        if remaining <= 0:
            break

        if len(block) > remaining:
            # Hypercube compression is lossy activation, not total erasure:
            # retain the highest-priority prefix that fits this pulse.
            fragment = block[:remaining]
            pieces.append(fragment)
            size += len(fragment)
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





def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for pulse/workload mathematics."""
    return max(1, (len(text) + 3) // 4)


def _repair_pulse_metrics(
    *,
    objective: str,
    failures: str,
    research: str,
    attempt: int,
    max_repairs: int,
    load: float = 0.0,
) -> PulseMetrics:
    """Project current repair state into Adrien D. Thomas Heartbeat metrics."""
    failure_present = bool(failures.strip())

    research_available = bool(
        research.strip()
        and "ONLINE_RESEARCH_UNAVAILABLE" not in research
    )

    lowered_failure = failures.casefold()

    contradiction = (
        0.95
        if "contradict" in lowered_failure
        else 0.0
    )

    progress = max(
        0.0,
        min(
            1.0,
            float(attempt) / max(1, max_repairs),
        ),
    )

    return PulseMetrics(
        observer=1.0,
        actor=0.80 if failure_present else 0.68,
        background=0.84 if research_available else 0.56,
        load=max(0.0, min(1.0, load)),
        prediction_error=0.88 if failure_present else 0.10,
        uncertainty=(
            0.62
            if failure_present
            else (0.42 if not research_available else 0.22)
        ),
        goal_urgency=0.35 + 0.55 * progress,
        meaningful_change=0.82 if failure_present else 0.20,
        contradiction=contradiction,
        evidence_quality=0.84 if research_available else 0.62,
        task_completion=0.12 + 0.18 * progress,
    )



def _engine_help_supports_prediction_limit(help_text: str) -> bool:
    """Detect bounded-generation support from the engine's advertised CLI."""
    for line in help_text.splitlines():
        tokens = line.replace(",", " ").replace("=", " ").split()

        if "-n" in tokens:
            return True

        if any(
            token == "--n-predict"
            or token.startswith("--n-predict=")
            for token in tokens
        ):
            return True

    return False


def _engine_supports_prediction_limit(engine: Path) -> bool:
    try:
        result = subprocess.run(
            [str(engine), "--help"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return _engine_help_supports_prediction_limit(
        (result.stdout or "") + "\\n" + (result.stderr or "")
    )

def _local_watchdog_seconds(
    *,
    prompt: str,
    requested_output_tokens: int,
    metrics: PulseMetrics,
) -> float:
    """Stall protection only. Never defines completion of thought."""
    observed_tps = max(
        0.1,
        float(os.getenv("GARVIS_LOCAL_OBSERVED_TPS", "3.0")),
    )

    recent_runtime = max(
        0.0,
        float(os.getenv("GARVIS_LOCAL_RECENT_RUNTIME", "0")),
    )

    maximum = max(
        60.0,
        float(os.getenv("GARVIS_HEARTBEAT_WATCHDOG_MAX", "900")),
    )

    watchdog = AdaptiveWatchdog(
        minimum_seconds=20.0,
        maximum_seconds=maximum,
        safety_margin=1.8,
    )

    return watchdog.estimate(
        prompt_tokens=_estimate_tokens(prompt),
        requested_output_tokens=requested_output_tokens,
        observed_tokens_per_second=observed_tps,
        device_load=metrics.load,
        recent_runtime_seconds=recent_runtime,
    )

def _local_edit_to_patch(repository: Path, response: str) -> str:
    """Convert one grounded local-model proposal into a deterministic Git diff."""
    clean = response.strip()

    if not clean:
        raise RuntimeError("local GARVIS returned an empty repair response")

    # Safe no-change results may contain a small amount of model formatting.
    normalized = re.sub(r"[^a-z_]+", "_", clean.casefold()).strip("_")
    if clean == "NO_PATCH_NEEDED" or normalized in {
        "no_patch_needed",
        "no_patch",
        "none",
    }:
        return "NO_PATCH_NEEDED"

    relative: str | None = None
    old: str | None = None
    new: str | None = None

    # Preferred machine protocol: one JSON object.
    json_candidates = [clean]

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        json_candidates.insert(0, fenced.group(1))

    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if 0 <= first_brace < last_brace:
        json_candidates.append(clean[first_brace:last_brace + 1])

    payload = None
    for candidate in json_candidates:
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(decoded, dict):
            payload = decoded
            break

    if payload is not None:
        action = str(payload.get("action", "")).strip().casefold()

        if action in {
            "none",
            "no_patch",
            "no_patch_needed",
            "noop",
        }:
            return "NO_PATCH_NEEDED"

        if action not in {"replace", "edit"}:
            raise RuntimeError(
                f"local GARVIS returned unsupported JSON action: {action!r}"
            )

        relative = str(payload.get("path", "")).strip()

        raw_old = payload.get("old")
        raw_new = payload.get("new")

        if not isinstance(raw_old, str) or not isinstance(raw_new, str):
            raise RuntimeError(
                "local GARVIS JSON edit requires string old/new fields"
            )

        old = raw_old
        new = raw_new

    else:
        # Backward-compatible parser for the previous exact-edit envelope.
        marker_index = clean.find("GARVIS_EDIT")
        if marker_index >= 0:
            clean = clean[marker_index:]

        match = re.fullmatch(
            r"GARVIS_EDIT\s*\n"
            r"PATH:\s*([^\n]+)\n"
            r"OLD:\s*\n<<<\n(.*?)\n>>>\n"
            r"NEW:\s*\n<<<\n(.*?)\n>>>\n"
            r"END\s*",
            clean,
            flags=re.DOTALL,
        )

        if match is None:
            excerpt = " ".join(response.strip().split())[:700]
            raise RuntimeError(
                "local GARVIS did not return a machine-readable repair; "
                f"response_excerpt={excerpt!r}"
            )

        relative = match.group(1).strip()
        old = match.group(2)
        new = match.group(3)

    assert relative is not None
    assert old is not None
    assert new is not None

    if (
        not relative
        or Path(relative).is_absolute()
        or relative.startswith("../")
        or "/../" in relative
    ):
        raise RuntimeError(f"invalid local edit path: {relative!r}")

    if relative in _BLOCKED_REPAIR_PATHS:
        raise RuntimeError(
            f"autonomous repair path is blocked: {relative}"
        )

    if is_protected_path(relative):
        raise RuntimeError(
            f"governance-protected path is blocked: {relative}"
        )

    if not relative.startswith(_ALLOWED_PREFIXES):
        raise RuntimeError(
            f"path outside autonomous repair scope: {relative}"
        )

    root = repository.resolve()
    target = (root / relative).resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"local edit escaped repository: {relative}"
        ) from exc

    if target.is_symlink() or not target.is_file():
        raise RuntimeError(
            f"local edit target is not a regular existing file: {relative}"
        )

    if not old:
        raise RuntimeError("local edit old text must not be empty")

    if old == new:
        raise RuntimeError("local edit does not change anything")

    for pattern in _SECRET_PATTERNS:
        if pattern.search(new):
            raise RuntimeError(
                "local edit replacement appears to contain credential material"
            )

    existing = target.read_text(encoding="utf-8")
    matches = existing.count(old)

    if matches != 1:
        raise RuntimeError(
            f"local edit old text must match exactly once in {relative}; "
            f"observed matches={matches}"
        )

    updated = existing.replace(old, new, 1)

    diff = "".join(
        difflib.unified_diff(
            existing.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )

    if not diff:
        raise RuntimeError("local edit generated an empty diff")

    patch = f"diff --git a/{relative} b/{relative}\n{diff}"
    _validate_patch(patch)
    return patch

def _request_patch_local(
    repository: Path,
    objective: str,
    failures: str,
    research: str,
    *,
    pulse_metrics: PulseMetrics | None = None,
) -> str:
    """Ask the on-device GARVIS brain for one structured repair proposal."""
    from garvis.local_language_runtime import (
        LocalRuntimeConfig,
        clean_model_output,
    )

    config = LocalRuntimeConfig.from_environment(repository)
    config.validate()

    raw_source = _context_bundle(
        repository,
        limit=5200,
    )

    load = min(
        1.0,
        _estimate_tokens(raw_source)
        / max(1, config.context_size),
    )

    metrics = pulse_metrics or _repair_pulse_metrics(
        objective=objective,
        failures=failures,
        research=research,
        attempt=1,
        max_repairs=1,
        load=load,
    )

    # Balloon compression:
    # increasing pressure narrows the ACTIVE semantic field.
    source_budget = max(
        1500,
        int(
            5200
            * (1.0 - 0.52 * metrics.pressure)
        ),
    )

    source = _context_bundle(
        repository,
        limit=source_budget,
    )

    failure_evidence = failures[-650:]
    research_evidence = research[-320:]

    # One pulse should emit a bounded machine decision,
    # not unlimited prose.
    output_budget = max(
        96,
        min(
            256,
            int(
                256
                - 104 * metrics.pressure
            ),
        ),
    )

    prompt = (
        "/no_think\n"
        "You are the local reasoning organ of GARVIS, architected by "
        "Adrien D. Thomas.\n"
        "Perform one bounded source-code repair analysis.\n"
        "Do not explain your answer.\n"
        "Do not output Markdown.\n"
        "Do not output Git diff syntax.\n"
        "Return exactly ONE JSON object.\n\n"
        "Allowed results:\n"
        '{"action":"none"}\n'
        "OR\n"
        '{"action":"replace","path":"src/garvis/file.py",'
        '"old":"EXACT EXISTING TEXT","new":"REPLACEMENT TEXT"}\n\n'
        "Rules:\n"
        "- old must be copied exactly from repository evidence.\n"
        "- old should be large enough to occur exactly once.\n"
        "- edit only src/garvis/** or tests/garvis/**.\n"
        "- never edit THANOS authorization, stage gates, GitHub maintenance, "
        "deployment controls, workflows, CODEOWNERS, or governance.\n"
        "- never include credentials or secrets.\n"
        "- never weaken validation.\n"
        "- if evidence does not justify a safe useful change, use action none.\n"
        "- external/model statements are candidate reasoning, not evidence.\n\n"
        "HYPERCUBE HEARTBEAT:\n"
        "RECEIVE -> SEGMENT -> PREDICT -> VERIFY -> SIMULATE -> PLAN -> "
        "OUTPUT -> FEEDBACK -> CONSOLIDATE\n\n"
        "OBJECTIVE:\n"
        + objective[:900]
        + "\n\nCURRENT FAILURE EVIDENCE:\n"
        + (failure_evidence or "none")
        + "\n\nEXTERNAL EVIDENCE:\n"
        + (research_evidence or "unavailable")
        + "\n\nREPOSITORY EVIDENCE:\n"
        + source
        + "\n\nHYPERCUBE PULSE:\n"
        + f"coherence={metrics.coherence:.4f}\n"
        + f"pressure={metrics.pressure:.4f}\n"
        + f"uncertainty={metrics.uncertainty:.4f}\n"
        + "\nJSON ONLY:\n"
    )

    watchdog_seconds = _local_watchdog_seconds(
        prompt=prompt,
        requested_output_tokens=output_budget,
        metrics=metrics,
    )

    print(
        f"LOCAL_PULSE_PRESSURE={metrics.pressure:.4f}"
    )
    print(
        f"ACTIVE_CONTEXT_CHARS={len(source)}"
    )
    print(
        f"LOCAL_OUTPUT_BUDGET={output_budget}"
    )
    print(
        f"WATCHDOG_SECONDS={watchdog_seconds:.1f}"
    )

    command = [
        str(config.engine),
        "-m",
        str(config.model),
        "-c",
        str(config.context_size),
        "-ngl",
        str(config.gpu_layers),
    ]

    if _engine_supports_prediction_limit(config.engine):
        command.extend(
            [
                "-n",
                str(output_budget),
            ]
        )
        print("LOCAL_PREDICTION_LIMIT=SUPPORTED")
    else:
        # The mathematical output budget remains part of pulse state.
        # This engine cannot enforce it directly, so the adaptive watchdog
        # remains process-stall protection.
        print(
            "LOCAL_PREDICTION_LIMIT=ENGINE_UNSUPPORTED"
        )

    try:
        result = subprocess.run(
            command,
            cwd=str(repository),
            input=prompt + "\n",
            text=True,
            capture_output=True,
            timeout=watchdog_seconds,
            check=False,
            env={
                **os.environ,
                "GARVIS_MEMORY_ENABLED": "0",
            },
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "PROCESS_STALLED: adaptive watchdog expired after "
            f"{watchdog_seconds:.1f}s; "
            "watchdog expiration is not THOUGHT_COMPLETE"
        ) from exc

    if result.returncode:
        detail = clean_model_output(
            result.stderr or result.stdout
        )
        raise RuntimeError(
            f"local GARVIS engine exited with code "
            f"{result.returncode}: {detail[-2000:]}"
        )

    response = clean_model_output(
        result.stdout
    ) or clean_model_output(result.stderr)

    if not response:
        raise RuntimeError(
            "local GARVIS returned an empty machine response"
        )

    return _local_edit_to_patch(repository, response)

def _request_patch(
    repository: Path,
    model: str,
    objective: str,
    failures: str,
    research: str,
    *,
    pulse_metrics: PulseMetrics | None = None,
) -> str:
    """Ask one reasoning organ for a bounded repair candidate."""
    if _provider_label(model) == "local":
        return _request_patch_local(
            repository,
            objective,
            failures,
            research,
            pulse_metrics=pulse_metrics,
        )

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
        if _provider_label(candidate) == "local":
            continue
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

    heartbeat_state = HeartbeatState()
    release_gate = ReleaseGate()

    for attempt in range(1, max_repairs + 1):
        pulse_metrics = _repair_pulse_metrics(
            objective=objective,
            failures=failures,
            research=research,
            attempt=attempt,
            max_repairs=max_repairs,
        )

        heartbeat_state = heartbeat_state.advance(
            pulse_metrics,
            dt=1.0,
        )

        boundary = detect_event_boundary(
            pulse_metrics
        )

        print(f"HEARTBEAT_ATTEMPT={attempt}")
        print(
            "HEARTBEAT_PERSPECTIVE="
            f"{heartbeat_state.perspective.value}"
        )
        print(
            "HEARTBEAT_PRESSURE="
            f"{pulse_metrics.pressure:.4f}"
        )
        print(
            "HEARTBEAT_COHERENCE="
            f"{pulse_metrics.coherence:.4f}"
        )
        print(
            "HEARTBEAT_REVOLUTIONS="
            f"{heartbeat_state.revolutions}"
        )
        print(
            "EVENT_BOUNDARY="
            + (
                boundary.reason
                if boundary.triggered
                else "none"
            )
        )
        print(
            "PHASE=RECEIVE_SEGMENT_PREDICT_VERIFY_SIMULATE_PLAN"
        )

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
                        pulse_metrics=pulse_metrics,
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

            release_metrics = PulseMetrics(
                observer=1.0,
                actor=1.0,
                background=1.0,
                load=0.20,
                prediction_error=0.0,
                uncertainty=0.0,
                goal_urgency=1.0,
                meaningful_change=1.0,
                contradiction=0.0,
                evidence_quality=1.0,
                task_completion=1.0,
            )

            if not release_gate.ready(
                release_metrics,
                deterministic_gates_passed=passed,
            ):
                raise RuntimeError(
                    "HEARTBEAT_RELEASE_BLOCKED: "
                    f"readiness="
                    f"{release_metrics.release_readiness:.4f}"
                )

            print(
                "HEARTBEAT_RELEASE_READINESS="
                f"{release_metrics.release_readiness:.4f}"
            )
            print("HEARTBEAT_RELEASE=READY")

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
