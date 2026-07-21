"""SQLite archival and memory storage for GARVIS."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import EvidenceStatus, MemoryKind, MemoryRecord


class NeuroStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    intended TEXT NOT NULL,
                    observed TEXT NOT NULL,
                    error_signal TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_session
                ON memories(session_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_memories_kind
                ON memories(kind, id DESC);
                """
            )

    def add_memory(
        self,
        *,
        session_id: str,
        kind: MemoryKind,
        status: EvidenceStatus,
        content: str,
        source: str,
        confidence: float,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        clean = content.strip()
        if not clean:
            raise ValueError("memory content must not be empty")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    session_id, kind, status, content, source,
                    confidence, importance, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind.value,
                    status.value,
                    clean,
                    source,
                    max(0.0, min(1.0, confidence)),
                    max(0.0, min(1.0, importance)),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def add_episode(self, session_id: str, user_text: str, assistant_text: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodes (session_id, user_text, assistant_text)
                VALUES (?, ?, ?)
                """,
                (session_id, user_text, assistant_text),
            )
            return int(cursor.lastrowid)

    def add_feedback(
        self,
        *,
        session_id: str,
        intended: str,
        observed: str,
        error_signal: str,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (
                    session_id, intended, observed, error_signal
                )
                VALUES (?, ?, ?, ?)
                """,
                (session_id, intended, observed, error_signal),
            )
            return int(cursor.lastrowid)

    def list_memories(
        self,
        *,
        session_id: str | None = None,
        limit: int = 1000,
    ) -> list[MemoryRecord]:
        query = """
            SELECT id, created_at, session_id, kind, status, content,
                   source, confidence, importance, metadata_json
            FROM memories
        """
        parameters: tuple[Any, ...]
        if session_id:
            query += " WHERE session_id IN (?, 'global')"
            parameters = (session_id,)
        else:
            parameters = ()
        query += " ORDER BY id DESC LIMIT ?"
        parameters += (max(1, limit),)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        result: list[MemoryRecord] = []
        for row in rows:
            result.append(
                MemoryRecord(
                    id=int(row["id"]),
                    created_at=str(row["created_at"]),
                    session_id=str(row["session_id"]),
                    kind=MemoryKind(str(row["kind"])),
                    status=EvidenceStatus(str(row["status"])),
                    content=str(row["content"]),
                    source=str(row["source"]),
                    confidence=float(row["confidence"]),
                    importance=float(row["importance"]),
                    metadata=json.loads(str(row["metadata_json"])),
                )
            )
        return result

    def count(self, table: str) -> int:
        if table not in {"memories", "episodes", "feedback"}:
            raise ValueError("unsupported table")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    def seed_identity(self, records: Iterable[tuple[str, str]]) -> None:
        existing = {
            memory.content
            for memory in self.list_memories(session_id="global", limit=500)
            if memory.kind is MemoryKind.IDENTITY
        }
        for source, content in records:
            if content in existing:
                continue
            self.add_memory(
                session_id="global",
                kind=MemoryKind.IDENTITY,
                status=EvidenceStatus.VERIFIED,
                content=content,
                source=source,
                confidence=1.0,
                importance=1.0,
            )
