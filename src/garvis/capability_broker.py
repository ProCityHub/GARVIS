"""Permission-gated capability broker for local GARVIS."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    session_id: str
    original_request: str
    research_query: str
    state: ApprovalState
    created_at: datetime
    expires_at: datetime

    def render(self) -> str:
        return (
            "GARVIS requests internet research permission.\n\n"
            "Purpose: Research this topic and return a sourced local answer\n"
            f"Search query leaving phone: {self.research_query}\n"
            "Other data leaving phone: Ordinary HTTPS request metadata only\n"
            "Estimated data: Usually under 2 MB; hard limits enforced\n"
            "Risk: Medium — public internet content may be inaccurate or malicious\n"
            "Access: One research task only\n"
            f"Expires: {self.expires_at.astimezone().strftime('%H:%M:%S')}\n\n"
            "Approve? [Y/N]"
        )


@dataclass(frozen=True)
class ApprovalResolution:
    request: ApprovalRequest
    approved: bool


_EXPLICIT = (
    re.compile(
        r"\b(?:you\s+may|i\s+authorize\s+you\s+to|permission\s+to)\s+"
        r"(?:use|access|search|browse|research|go\s+on)\b.{0,60}\b"
        r"(?:internet|web|online|phone\s+data|mobile\s+data)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:search|research|browse|look\s+up|check)\b.{0,50}\b"
        r"(?:on|using|through)\s+(?:the\s+)?(?:internet|web|online)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:go\s+online|use\s+(?:the\s+)?internet|use\s+(?:my\s+)?"
        r"(?:phone|mobile)\s+data)\b",
        re.IGNORECASE,
    ),
)
_RESEARCH = (
    re.compile(r"\b(?:search|browse|research|look\s+up|find\s+online)\b", re.I),
    re.compile(
        r"\b(?:latest|today|current|currently|news|weather|forecast|live\s+score|"
        r"stock\s+price|exchange\s+rate)\b",
        re.I,
    ),
)
_APPROVE = {"y", "yes", "approve", "approved"}
_DENY = {"n", "no", "deny", "denied", "cancel"}


def has_explicit_network_authorization(message: str) -> bool:
    clean = _clean(message)
    return any(pattern.search(clean) for pattern in _EXPLICIT)


def appears_to_require_research(message: str) -> bool:
    clean = _clean(message)
    return any(pattern.search(clean) for pattern in _RESEARCH)


def extract_research_query(message: str) -> str:
    clean = _clean(message)
    patterns = (
        r"^(?:garvis[,:\s]+)?(?:you\s+may\s+)?(?:please\s+)?"
        r"(?:use|access|search|browse|research|go\s+on)\s+"
        r"(?:the\s+)?(?:internet|web|online|phone\s+data|mobile\s+data)"
        r"(?:\s+to)?\s*"
        r"(?:(?:search|research|browse|look\s+up|check)\s+)?",
        r"^(?:garvis[,:\s]+)?(?:please\s+)?"
        r"(?:search|research|browse|look\s+up|check)\s+"
        r"(?:on|using|through)\s+(?:the\s+)?(?:internet|web|online)"
        r"(?:\s+for)?\s*",
    )
    for pattern in patterns:
        updated = re.sub(pattern, "", clean, flags=re.I)
        if updated != clean and updated:
            return updated[:500]
    return clean[:500]


class ApprovalStore:
    def __init__(self, db_path: Path | None = None) -> None:
        default = Path.home() / ".garvis" / "capability_broker.db"
        self.db_path = (
            db_path or Path(os.getenv("GARVIS_CAPABILITY_DB", str(default)))
        ).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS approvals(
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                original_request TEXT NOT NULL,
                research_query TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS capability_audit(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                request_id TEXT,
                session_id TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ApprovalStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=str(row["request_id"]),
            session_id=str(row["session_id"]),
            original_request=str(row["original_request"]),
            research_query=str(row["research_query"]),
            state=ApprovalState(str(row["state"])),
            created_at=_parse(str(row["created_at"])),
            expires_at=_parse(str(row["expires_at"])),
        )

    def audit(
        self,
        event: str,
        *,
        session_id: str,
        request_id: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO capability_audit(event,request_id,session_id,detail_json,created_at) "
            "VALUES(?,?,?,?,?)",
            (event, request_id, session_id, json.dumps(detail or {}, sort_keys=True), _iso(_now())),
        )
        self.connection.commit()

    def create(
        self,
        original_request: str,
        research_query: str,
        *,
        session_id: str = "default",
        ttl_minutes: int = 10,
    ) -> ApprovalRequest:
        self.expire(session_id=session_id)
        now = _now()
        request = ApprovalRequest(
            uuid.uuid4().hex,
            session_id,
            _clean(original_request),
            _clean(research_query),
            ApprovalState.PENDING,
            now,
            now + timedelta(minutes=ttl_minutes),
        )
        self.connection.execute(
            "INSERT INTO approvals(request_id,session_id,original_request,research_query,state,"
            "created_at,expires_at) VALUES(?,?,?,?,?,?,?)",
            (
                request.request_id,
                request.session_id,
                request.original_request,
                request.research_query,
                request.state.value,
                _iso(request.created_at),
                _iso(request.expires_at),
            ),
        )
        self.connection.commit()
        self.audit(
            "approval_requested",
            session_id=session_id,
            request_id=request.request_id,
            detail={"query": request.research_query},
        )
        return request

    def pending(self, *, session_id: str = "default") -> ApprovalRequest | None:
        self.expire(session_id=session_id)
        row = self.connection.execute(
            "SELECT * FROM approvals WHERE session_id=? AND state=? "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id, ApprovalState.PENDING.value),
        ).fetchone()
        return self._row(row) if row is not None else None

    def expire(self, *, session_id: str = "default") -> int:
        rows = self.connection.execute(
            "SELECT request_id FROM approvals WHERE session_id=? AND state=? AND expires_at<=?",
            (session_id, ApprovalState.PENDING.value, _iso(_now())),
        ).fetchall()
        if not rows:
            return 0
        ids = [str(row["request_id"]) for row in rows]
        marks = ",".join("?" for _ in ids)
        self.connection.execute(
            f"UPDATE approvals SET state=?,resolved_at=? WHERE request_id IN ({marks})",
            (ApprovalState.EXPIRED.value, _iso(_now()), *ids),
        )
        self.connection.commit()
        return len(ids)

    def resolve(self, message: str, *, session_id: str = "default") -> ApprovalResolution | None:
        answer = _clean(message).casefold()
        if answer not in _APPROVE | _DENY:
            return None
        request = self.pending(session_id=session_id)
        if request is None:
            return None
        approved = answer in _APPROVE
        state = ApprovalState.APPROVED if approved else ApprovalState.DENIED
        self.connection.execute(
            "UPDATE approvals SET state=?,resolved_at=? WHERE request_id=? AND state=?",
            (state.value, _iso(_now()), request.request_id, ApprovalState.PENDING.value),
        )
        self.connection.commit()
        resolved = replace(request, state=state)
        self.audit(
            "approval_granted" if approved else "approval_denied",
            session_id=session_id,
            request_id=request.request_id,
            detail={"answer": answer},
        )
        return ApprovalResolution(resolved, approved)

    def recent_audit(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM capability_audit ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "event": str(row["event"]),
                "request_id": row["request_id"],
                "session_id": str(row["session_id"]),
                "detail": json.loads(str(row["detail_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]
