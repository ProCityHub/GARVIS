#!/usr/bin/env python3
"""Quarantine derived self-echo memories while preserving exact episodes."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--session", default="neuro-0.1")
    result.add_argument("--db", type=Path, default=None)
    return result


def main() -> None:
    args = parser().parse_args()
    home = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
    db = (args.db or home / "neurocognitive.db").expanduser().resolve()

    if not db.is_file():
        print(f"No neurocognitive database found at {db}")
        return

    with sqlite3.connect(db) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_quarantine (
                quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,
                quarantined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        rows = connection.execute(
            """
            SELECT id, created_at, session_id, kind, status, content, source,
                   confidence, importance, metadata_json
            FROM memories
            WHERE session_id = ?
              AND kind IN ('episode', 'dream')
            """,
            (args.session,),
        ).fetchall()

        for row in rows:
            connection.execute(
                """
                INSERT INTO memory_quarantine (
                    original_id, reason, created_at, session_id, kind, status,
                    content, source, confidence, importance, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row[0],
                    "pre-fix derived memory could contain multiline-split or self-echo content",
                    *row[1:],
                ),
            )

        connection.execute(
            """
            DELETE FROM memories
            WHERE session_id = ?
              AND kind IN ('episode', 'dream')
            """,
            (args.session,),
        )

    print(f"Quarantined {len(rows)} derived memories from session {args.session}.")
    print("Exact user/GARVIS episode pairs remain preserved in the episodes table.")
    print(f"Database: {db}")


if __name__ == "__main__":
    main()
