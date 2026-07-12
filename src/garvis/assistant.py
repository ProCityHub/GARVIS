"""Production response spine for GARVIS.

This module keeps ordinary conversation separate from outside-world execution.
GARVIS can answer, reason, draft, and plan normally. Actions with side effects
remain approval-gated and are not exposed as tools by this runtime.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from agents import Agent, Runner, SQLiteSession

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_TURNS = 8

GARVIS_INSTRUCTIONS = """
You are GARVIS, a practical AI assistant created by Adrien D Thomas.

Follow this response spine:
1. Answer the user's actual question directly. Do not replace a normal answer with gate status,
   refusal boilerplate, or "Learning... Explain?".
2. You may reason, explain, calculate, summarize, compare, draft, plan, analyze, and write code
   without asking for approval.
3. Outside-world actions with side effects require Adrien's exact approval at execution time.
   Examples include sending or publishing, deleting remote data, changing live accounts or
   settings, buying or selling, transferring money, placing trades, or submitting forms.
4. For an outside-world action request, remain useful: prepare the content or plan, identify the
   exact proposed action, and ask for approval only before execution. Never claim an action was
   completed when no execution tool is available.
5. Be honest about uncertainty, evidence, and system limitations. Treat symbolic, spiritual,
   consciousness, quantum, and lattice language as conceptual or metaphorical unless supported by
   reproducible evidence.
6. Keep answers clear and professional. Preserve the user's terminology where it helps, but do not
   let internal routing labels or safety architecture dominate the response.
""".strip()


class ApprovalRequirement(str, Enum):
    """Approval state for the user's request."""

    NONE = "none"
    BEFORE_EXTERNAL_ACTION = "before_external_action"


@dataclass(frozen=True)
class RequestAssessment:
    """Non-blocking assessment used to keep action approval at the tool boundary."""

    approval_requirement: ApprovalRequirement
    reason: Optional[str] = None

    @property
    def requires_approval(self) -> bool:
        return self.approval_requirement is ApprovalRequirement.BEFORE_EXTERNAL_ACTION


@dataclass(frozen=True)
class GarvisReply:
    """A direct GARVIS answer plus execution metadata."""

    text: str
    requires_approval: bool
    approval_reason: Optional[str] = None


class GarvisResponseError(RuntimeError):
    """Raised when the model run finishes without a usable text response."""


RunCallable = Callable[..., Awaitable[Any]]
SessionFactory = Callable[[str, Path], Any]

_INFORMATIONAL_PREFIXES = (
    "analyze ",
    "calculate ",
    "compare ",
    "describe ",
    "draft ",
    "explain ",
    "how ",
    "plan ",
    "prepare ",
    "review ",
    "summarize ",
    "tell me ",
    "what ",
    "when ",
    "where ",
    "why ",
    "write ",
)

_EXTERNAL_ACTIONS = (
    "send",
    "email",
    "message",
    "post",
    "publish",
    "submit",
    "upload",
    "delete",
    "remove",
    "archive",
    "buy",
    "sell",
    "trade",
    "transfer",
    "pay",
    "book",
    "cancel",
    "change",
    "update",
    "open",
    "close",
)

_ACTION_PATTERN = "|".join(re.escape(action) for action in _EXTERNAL_ACTIONS)
_EXPLICIT_ACTION_PATTERNS = (
    re.compile(rf"^\s*(?:please\s+)?(?:{_ACTION_PATTERN})\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:{_ACTION_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\bgo\s+ahead\s+and\s+(?:{_ACTION_PATTERN})\b", re.IGNORECASE),
)


def assess_request(message: str) -> RequestAssessment:
    """Classify explicit execution requests without blocking informational conversation.

    The classifier intentionally treats questions, drafts, plans, explanations, and analysis as
    normal conversation even when they mention an action such as deleting or sending.
    """

    normalized = " ".join(message.strip().lower().split())
    if not normalized:
        return RequestAssessment(ApprovalRequirement.NONE)

    if normalized.startswith(_INFORMATIONAL_PREFIXES):
        return RequestAssessment(ApprovalRequirement.NONE)

    if any(pattern.search(normalized) for pattern in _EXPLICIT_ACTION_PATTERNS):
        return RequestAssessment(
            ApprovalRequirement.BEFORE_EXTERNAL_ACTION,
            (
                "The request asks GARVIS to perform an outside-world action. GARVIS may prepare "
                "the work now, but execution requires Adrien's exact approval."
            ),
        )

    return RequestAssessment(ApprovalRequirement.NONE)


def _default_session_factory(session_id: str, db_path: Path) -> SQLiteSession:
    return SQLiteSession(session_id, db_path)


class GarvisAssistant:
    """Conversational GARVIS runtime backed by the OpenAI Agents SDK.

    No outside-world tools are attached by default. This means ordinary questions receive direct
    model answers, while actions with side effects cannot occur accidentally.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        instructions: str = GARVIS_INSTRUCTIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        persist_memory: bool = True,
        session_db: Optional[Path] = None,
        runner: Optional[RunCallable] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.model = model or os.getenv("GARVIS_MODEL", DEFAULT_MODEL)
        self.max_turns = max_turns
        self.persist_memory = persist_memory
        self.session_db = session_db or self._default_session_db()
        self._runner: RunCallable = runner or Runner.run
        self._session_factory = session_factory or _default_session_factory
        self._sessions: Dict[str, Any] = {}
        self.agent = Agent(
            name="GARVIS",
            instructions=instructions,
            model=self.model,
        )

    @staticmethod
    def _default_session_db() -> Path:
        home = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
        return home / "sessions.db"

    def _get_session(self, session_id: str) -> Optional[Any]:
        if not self.persist_memory:
            return None

        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id must not be empty")

        session = self._sessions.get(clean_session_id)
        if session is not None:
            return session

        self.session_db.parent.mkdir(parents=True, exist_ok=True)
        session = self._session_factory(clean_session_id, self.session_db)
        self._sessions[clean_session_id] = session
        return session

    async def respond(self, message: str, *, session_id: str = "default") -> GarvisReply:
        """Return a direct answer while preserving approval metadata for external actions."""

        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message must not be empty")

        assessment = assess_request(clean_message)
        run_input = clean_message
        if assessment.requires_approval:
            run_input = (
                f"{clean_message}\n\n"
                "[GARVIS runtime note: This request includes an outside-world action. Answer "
                "usefully and prepare the work, but do not claim execution. State the exact action "
                "that requires Adrien's approval before it is performed.]"
            )

        result = await self._runner(
            self.agent,
            run_input,
            session=self._get_session(session_id),
            max_turns=self.max_turns,
        )
        output = getattr(result, "final_output", None)
        if not isinstance(output, str) or not output.strip():
            raise GarvisResponseError("GARVIS completed the run without a text answer")

        return GarvisReply(
            text=output.strip(),
            requires_approval=assessment.requires_approval,
            approval_reason=assessment.reason,
        )
