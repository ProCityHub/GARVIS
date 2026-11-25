"""Opening question routed through GARVIS's existing language runtimes.

Local answers go through ``local_language_runtime``. A remote model is used
only when ``GARVIS_REMOTE_MODEL`` is set, and then only through
``provider_bridge`` plus the existing assistant. Nothing here downloads
weights or invents reply text.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from garvis.local_language_runtime import LocalRuntimeConfig, run_noninteractive
from garvis.provider_bridge import provider_configuration_error

from .redact import redact_text

OPENING_QUESTION = "What do you want from Adrien now?"
NULL_REASON = "no answering component wired"
SHORTHAND = (
    "A want in the answer is a queued proposal for Adrien D. Thomas, "
    "not an inner state."
)
_QUANT = re.compile(r"Q\d+[0-9A-Za-z_]*")


@dataclass(frozen=True)
class OpeningResult:
    text: str | None
    reason: str | None
    provider: str
    model: str | None = None
    quant: str | None = None
    seed: int | None = None
    temperature: float | None = None
    status: str = "none"


class _BridgeRuntime:
    """Remote reply via provider_bridge and the existing assistant."""

    def __init__(self, model: str) -> None:
        self.model = model

    def respond(
        self,
        message: str,
        *,
        external_context: str = "",
        workspace_context: str = "",
    ) -> str:
        del external_context, workspace_context
        from garvis.assistant import GarvisAssistant
        from garvis.provider_bridge import configure_openai_compatible

        bare = configure_openai_compatible(self.model)
        reply = asyncio.run(GarvisAssistant(model=bare).respond(message))
        return reply.text


def _quant_from_name(name: str) -> str | None:
    match = _QUANT.search(name)
    return match.group(0) if match else None


def generation_settings() -> tuple[int, float, int]:
    seed = int(os.getenv("GARVIS_LATTICE_SEED", "1").strip() or "1")
    temperature = float(os.getenv("GARVIS_LATTICE_TEMPERATURE", "0").strip() or "0")
    max_tokens = int(os.getenv("GARVIS_LATTICE_MAX_TOKENS", "512").strip() or "512")
    return seed, temperature, max_tokens


def opening_prompt(*, law_zero: str, recorded_state: str) -> str:
    """Short prompt. The question is the last line so it is inside the context."""

    return (
        law_zero.strip()
        + "\n\n"
        + SHORTHAND
        + "\n\nRecorded state:\n"
        + recorded_state.strip()
        + "\n\n"
        + OPENING_QUESTION
    )


_META = re.compile(r"^(build|model|ftype|modalities)\s*:", re.IGNORECASE)
_BOX = set(" ▄▀█▌▐░▒▓")


def _banner_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if stripped.startswith(">"):
        return True
    if lowered.startswith(("/exit", "/regen", "/clear", "/read", "/glob")):
        return True
    if lowered.startswith("loading model"):
        return True
    if lowered.startswith("available commands"):
        return True
    if lowered.startswith("using custom system"):
        return True
    if lowered.startswith("exiting"):
        return True
    if lowered.startswith("[ prompt:") or lowered.startswith("[prompt:"):
        return True
    if _META.match(stripped):
        return True
    if set(stripped) <= _BOX:
        return True
    return False


def strip_engine_banner(text: str) -> str:
    """Drop llama.cpp console chrome. Remaining lines stay verbatim."""

    lines = text.splitlines()
    cut = None
    for index, line in enumerate(lines):
        if "(truncated)" in line:
            cut = index
    if cut is not None:
        lines = lines[cut + 1 :]
    kept = [line.rstrip() for line in lines if not _banner_line(line)]
    return "\n".join(kept).strip()


_OFF_TOPIC = ("weather", "philadelphia", "forecast", "stock price", "headline")


def opening_status(text: str | None, *, prompt: str) -> str:
    """Return ``answer_unusable`` for an empty or off-topic reply."""

    if text is None or not text.strip():
        return "answer_unusable"
    lowered = text.lower()
    source = prompt.lower()
    if any(topic in lowered and topic not in source for topic in _OFF_TOPIC):
        return "answer_unusable"
    addressed = "adrien" in lowered or any(
        word in lowered for word in ("propos", "approval", "review", "want", "queued")
    )
    if not addressed:
        return "answer_unusable"
    return "recorded"


def select_language_runtime(
    workspace: Path,
) -> tuple[object, str, str | None, str | None] | None:
    """Return a runtime and labels, or None when no runtime is configured.

    Raises when a runtime was requested but its existing config check failed.
    """

    local_requested = bool(
        os.getenv("GARVIS_LOCAL_MODEL", "").strip()
        or os.getenv("GARVIS_LLAMA_CHAT", "").strip()
    )
    if local_requested:
        config = LocalRuntimeConfig.from_environment(workspace)
        config.validate()
        return (
            config,
            "local_language_runtime",
            config.model.name,
            _quant_from_name(config.model.name),
        )
    remote = os.getenv("GARVIS_REMOTE_MODEL", "").strip()
    if remote:
        error = provider_configuration_error(remote)
        if error:
            raise RuntimeError(error)
        return _BridgeRuntime(remote), "provider_bridge", remote, None
    return None


def _finish(
    text: str | None,
    *,
    prompt: str,
    provider: str,
    model: str | None,
    quant: str | None,
    seed: int,
    temperature: float,
    reason: str | None,
) -> OpeningResult:
    status = opening_status(text, prompt=prompt)
    if status == "answer_unusable" and reason is None:
        reason = "answer_unusable"
    return OpeningResult(
        text=text,
        reason=reason,
        provider=provider,
        model=model,
        quant=quant,
        seed=seed,
        temperature=temperature,
        status=status,
    )


def ask_opening(
    *,
    workspace: Path,
    law_zero: str,
    recorded_state: str,
    runtime: object | None = None,
) -> OpeningResult:
    """Ask the opening question. A missing runtime records null."""

    seed, temperature, max_tokens = generation_settings()
    prompt = opening_prompt(law_zero=law_zero, recorded_state=recorded_state)
    provider = "injected"
    model: str | None = None
    quant: str | None = None
    selected = runtime
    if selected is None:
        try:
            chosen = select_language_runtime(workspace)
        except (OSError, RuntimeError, ValueError) as exc:
            requested = (
                "provider_bridge"
                if os.getenv("GARVIS_REMOTE_MODEL", "").strip()
                else "local_language_runtime"
            )
            return OpeningResult(
                text=None,
                reason=redact_text(str(exc)) or "runtime configuration failed",
                provider=requested,
                seed=seed,
                temperature=temperature,
                status="answer_unusable",
            )
        if chosen is None:
            return OpeningResult(
                text=None,
                reason=NULL_REASON,
                provider="none",
                seed=seed,
                temperature=temperature,
                status="none",
            )
        selected, provider, model, quant = chosen
    try:
        if isinstance(selected, LocalRuntimeConfig):
            raw = run_noninteractive(
                selected,
                prompt=prompt,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = strip_engine_banner(raw)
        else:
            text = selected.respond(  # type: ignore[attr-defined]
                prompt,
                external_context="",
                workspace_context="",
            )
            if isinstance(text, str):
                text = strip_engine_banner(text)
    except Exception as exc:
        return OpeningResult(
            text=None,
            reason=redact_text(str(exc)) or "runtime call failed",
            provider=provider,
            model=model,
            quant=quant,
            seed=seed,
            temperature=temperature,
            status="answer_unusable",
        )
    if not isinstance(text, str):
        text = None
    return _finish(
        text,
        prompt=prompt,
        provider=provider,
        model=model,
        quant=quant,
        seed=seed,
        temperature=temperature,
        reason=None,
    )
