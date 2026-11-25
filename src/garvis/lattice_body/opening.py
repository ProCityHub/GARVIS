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

from garvis.local_language_runtime import LocalLanguageRuntime, LocalRuntimeConfig
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
        runtime = LocalLanguageRuntime(config, repository_root=workspace)
        return (
            runtime,
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


def ask_opening(
    *,
    workspace: Path,
    law_zero: str,
    recorded_state: str,
    runtime: object | None = None,
) -> OpeningResult:
    """Ask the opening question. A missing runtime records null."""

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
            )
        if chosen is None:
            return OpeningResult(
                text=None,
                reason=NULL_REASON,
                provider="none",
            )
        selected, provider, model, quant = chosen
    message = "\n\n".join(
        part
        for part in (
            OPENING_QUESTION,
            SHORTHAND,
            law_zero.strip(),
            recorded_state.strip(),
        )
        if part
    )
    try:
        text = selected.respond(  # type: ignore[attr-defined]
            message,
            external_context="",
            workspace_context="",
        )
    except Exception as exc:
        return OpeningResult(
            text=None,
            reason=redact_text(str(exc)) or "runtime call failed",
            provider=provider,
            model=model,
            quant=quant,
        )
    if not isinstance(text, str) or not text.strip():
        return OpeningResult(
            text=None,
            reason="runtime returned an empty answer",
            provider=provider,
            model=model,
            quant=quant,
        )
    return OpeningResult(
        text=text,
        reason=None,
        provider=provider,
        model=model,
        quant=quant,
    )
