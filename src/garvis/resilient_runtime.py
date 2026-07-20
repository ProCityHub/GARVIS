"""Bounded, crash-resilient conversational runtime for GARVIS."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from tools.garvis_resilience import SessionLedger, build_context, call_with_retry

from .assistant import GARVIS_INSTRUCTIONS, GarvisReply, assess_request
from .repository_context import ground_message, should_ground_repository

Message = Dict[str, str]
BuildContextCallable = Callable[[str, Any], List[Message]]
CallModelCallable = Callable[[Any, str, List[Message]], str]

_CURRENT_MESSAGE_MARKER = "ADRIEN'S CURRENT MESSAGE:"


def split_wrapped_prompt(prompt: str) -> Tuple[str, str]:
    """Separate local constitutional context from Adrien's current message.

    The historical Termux wrapper prepended its entire constitutional context
    to every message. That context may still be supplied to the model, but it
    must never be appended repeatedly to the persistent conversation ledger.
    """

    clean_prompt = prompt.strip()
    if _CURRENT_MESSAGE_MARKER not in clean_prompt:
        return "", clean_prompt

    context, marker, current_message = clean_prompt.rpartition(_CURRENT_MESSAGE_MARKER)
    if not marker:
        return "", clean_prompt

    return context.strip(), current_message.strip()


def _replace_latest_user_message(messages: List[Message], content: str) -> None:
    """Replace only the current model-facing user turn."""

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            messages[index] = {"role": "user", "content": content}
            return

    messages.append({"role": "user", "content": content})


class ResilientGarvisRuntime:
    """GARVIS runtime using an append-only ledger and bounded model context."""

    def __init__(
        self,
        *,
        model: str,
        session_name: str,
        repository_root: Optional[Path] = None,
        client: Optional[Any] = None,
        ledger: Optional[Any] = None,
        build_messages: BuildContextCallable = build_context,
        call_model: CallModelCallable = call_with_retry,
    ) -> None:
        clean_session = session_name.strip()
        if not clean_session:
            raise ValueError("session_name must not be empty")

        self.model = model
        self.session_name = clean_session
        self.repository_root = repository_root or Path.cwd()

        # The resilience module owns retry policy. Disable nested SDK retries.
        self.client = client or OpenAI(max_retries=0)
        self.ledger = ledger or SessionLedger(clean_session)
        self._build_messages = build_messages
        self._call_model = call_model

    async def respond(
        self,
        message: str,
        *,
        session_id: str = "default",
    ) -> GarvisReply:
        """Persist input, build bounded context, call safely, then persist reply."""

        del session_id  # Interface compatibility with GarvisAssistant.

        system_extension, user_text = split_wrapped_prompt(message)
        if not user_text:
            raise ValueError("message must not be empty")

        assessment = assess_request(user_text)

        # Integration point 2: persist immediately after input is read.
        # If the process dies after this line, Adrien's message survives.
        self.ledger.append("user", user_text)

        system_prompt = GARVIS_INSTRUCTIONS
        if system_extension:
            system_prompt = (
                f"{system_prompt}\n\n"
                "LOCAL TERMUX CONSTITUTIONAL CONTEXT:\n"
                f"{system_extension}"
            )

        # Integration point 3: only the bounded recent ledger window is sent.
        messages = self._build_messages(system_prompt, self.ledger)

        model_user_text = user_text

        # Repository evidence is supplied only to the current model request.
        # It is not copied into the durable ledger.
        if should_ground_repository(user_text):
            model_user_text = ground_message(user_text, self.repository_root)

        if assessment.requires_approval:
            model_user_text = (
                f"{model_user_text}\n\n"
                "[GARVIS runtime note: This request includes an outside-world "
                "action. Prepare useful work, but do not claim execution. State "
                "the exact action requiring Adrien's approval.]"
            )

        _replace_latest_user_message(messages, model_user_text)

        # Integration point 4: transient failures use bounded retry/backoff.
        reply_text = await asyncio.to_thread(
            self._call_model,
            self.client,
            self.model,
            messages,
        )

        if not isinstance(reply_text, str) or not reply_text.strip():
            raise RuntimeError("GARVIS completed the request without a text reply")

        clean_reply = reply_text.strip()

        # Integration point 5: persist the completed reply before printing it.
        self.ledger.append("assistant", clean_reply)

        return GarvisReply(
            text=clean_reply,
            requires_approval=assessment.requires_approval,
            approval_reason=assessment.reason,
        )
