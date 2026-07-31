"""Evidence-aware, human-inspired memory lifecycle for local GARVIS.

This is an engineering model inspired by retention, retrieval practice,
consolidation, interference, and forgetting. It is not a biological brain
simulation and does not imply consciousness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence


_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "i", "if", "in", "is", "it", "its", "my",
    "not", "of", "on", "or", "our", "so", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _digest(text: str) -> str:
    return hashlib.sha256(_clean(text).casefold().encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


class MemoryKind(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CORE = "core"
    TRACE = "trace"


class MemoryState(str, Enum):
    ACTIVE = "active"
    CONSOLIDATED = "consolidated"
    LATENT = "latent"
    TRACE = "trace"
    FORGOTTEN = "forgotten"


class EvidenceStatus(str, Enum):
    USER_SUPPLIED = "user_supplied"
    PROVISIONAL = "provisional_claim"
    EVIDENCE_SUPPORTED = "evidence_supported"
    VERIFIED = "verified"
    MODEL_GENERATED = "model_generated_unverified"


@dataclass(frozen=True)
class MemoryPolicy:
    working_half_life_hours: float = 8.0
    episodic_half_life_hours: float = 720.0
    semantic_half_life_hours: float = 4320.0
    procedural_half_life_hours: float = 8760.0
    core_half_life_hours: float = 87600.0
    consolidate_threshold: float = 0.62
    latent_threshold: float = 0.28
    trace_threshold: float = 0.12
    trace_min_age_hours: float = 168.0
    prompt_budget_chars: int = 1600
    max_recall_items: int = 8
    maintenance_interval: int = 12
    model_output_max_chars: int = 2400

    def half_life(self, kind: MemoryKind) -> float:
        return {
            MemoryKind.WORKING: self.working_half_life_hours,
            MemoryKind.EPISODIC: self.episodic_half_life_hours,
            MemoryKind.SEMANTIC: self.semantic_half_life_hours,
            MemoryKind.PROCEDURAL: self.procedural_half_life_hours,
            MemoryKind.CORE: self.core_half_life_hours,
            MemoryKind.TRACE: self.core_half_life_hours,
        }[kind]

    @classmethod
    def from_json(cls, path: Path) -> "MemoryPolicy":
        data = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    session_id: str
    kind: MemoryKind
    state: MemoryState
    evidence_status: EvidenceStatus
    content: str
    trace_hint: str
    source: str
    destination: str
    tags: tuple[str, ...]
    salience: float
    confidence: float
    arousal: float
    repetition_count: int
    retrieval_count: int
    protected: bool
    created_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True)
class RecallResult:
    memory: MemoryRecord
    relevance: float
    retention: float
    score: float


@dataclass(frozen=True)
class MaintenanceAction:
    memory_id: int
    previous_state: MemoryState
    next_state: MemoryState
    retention: float
    reason: str


@dataclass(frozen=True)
class MaintenanceReport:
    applied: bool
    scanned: int
    actions: tuple[MaintenanceAction, ...]


def retention_score(
    memory: MemoryRecord,
    *,
    now: Optional[datetime] = None,
    policy: Optional[MemoryPolicy] = None,
) -> float:
    policy = policy or MemoryPolicy()
    current = now or _now()
    age_hours = max(
        0.0,
        (current - memory.last_seen_at).total_seconds() / 3600.0,
    )
    reinforcement = 1.0 + 0.40 * math.log1p(
        memory.repetition_count + memory.retrieval_count
    )
    importance = (
        (0.35 + 0.65 * memory.salience)
        * (0.55 + 0.45 * memory.confidence)
        * (0.90 + 0.20 * memory.arousal)
    )
    half_life = max(0.01, policy.half_life(memory.kind) * reinforcement * importance)
    score = 2.0 ** (-age_hours / half_life)
    if memory.protected or memory.kind is MemoryKind.CORE:
        score = max(score, 0.50)
    if memory.state is MemoryState.TRACE:
        score = min(score, 0.10)
    if memory.state is MemoryState.FORGOTTEN:
        return 0.0
    return _clamp(score)


def lexical_relevance(query: str, memory: MemoryRecord) -> float:
    query_tokens = _tokens(query)
    memory_tokens = _tokens(memory.content or memory.trace_hint)
    if not query_tokens or not memory_tokens:
        return 0.0
    overlap = len(query_tokens & memory_tokens)
    union = len(query_tokens | memory_tokens)
    return _clamp(
        0.60 * (overlap / len(query_tokens))
        + 0.40 * (overlap / union if union else 0.0)
    )


def _make_trace(content: str, tags: Iterable[str], destination: str) -> str:
    counts: dict[str, int] = {}
    for token in _tokens(content):
        counts[token] = counts.get(token, 0) + 1
    keywords = sorted(counts, key=lambda item: (-counts[item], item))[:8]
    parts = [f"destination={destination}"]
    clean_tags = sorted({_clean(tag) for tag in tags if _clean(tag)})[:6]
    if clean_tags:
        parts.append("tags=" + ",".join(clean_tags))
    if keywords:
        parts.append("keywords=" + ",".join(keywords))
    return "; ".join(parts)[:240]


class MemoryStore:
    def __init__(
        self,
        db_path: Path,
        policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.db_path = db_path.expanduser()
        self.policy = policy or MemoryPolicy()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    @classmethod
    def from_environment(cls) -> "MemoryStore":
        db_path = Path(
            os.getenv(
                "GARVIS_MEMORY_DB",
                str(Path.home() / ".garvis" / "memory_lifecycle.db"),
            )
        )
        policy_path = os.getenv("GARVIS_MEMORY_POLICY", "").strip()
        policy = (
            MemoryPolicy.from_json(Path(policy_path).expanduser())
            if policy_path
            else MemoryPolicy()
        )
        return cls(db_path, policy)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                content TEXT NOT NULL,
                trace_hint TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL,
                destination TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                salience REAL NOT NULL,
                confidence REAL NOT NULL,
                arousal REAL NOT NULL,
                repetition_count INTEGER NOT NULL,
                retrieval_count INTEGER NOT NULL,
                protected INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(session_id, kind, content_hash)
            );
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _row(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            kind=MemoryKind(str(row["kind"])),
            state=MemoryState(str(row["state"])),
            evidence_status=EvidenceStatus(str(row["evidence_status"])),
            content=str(row["content"]),
            trace_hint=str(row["trace_hint"]),
            source=str(row["source"]),
            destination=str(row["destination"]),
            tags=tuple(json.loads(str(row["tags_json"]))),
            salience=float(row["salience"]),
            confidence=float(row["confidence"]),
            arousal=float(row["arousal"]),
            repetition_count=int(row["repetition_count"]),
            retrieval_count=int(row["retrieval_count"]),
            protected=bool(row["protected"]),
            created_at=_parse(str(row["created_at"])),
            last_seen_at=_parse(str(row["last_seen_at"])),
        )

    def get(self, memory_id: int) -> MemoryRecord:
        row = self.connection.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")
        return self._row(row)

    def remember(
        self,
        content: str,
        *,
        session_id: str = "default",
        kind: MemoryKind = MemoryKind.EPISODIC,
        evidence_status: EvidenceStatus = EvidenceStatus.USER_SUPPLIED,
        source: str = "user_input",
        destination: str = "general_dialogue",
        tags: Iterable[str] = (),
        salience: float = 0.50,
        confidence: float = 0.50,
        arousal: float = 0.0,
        protected: bool = False,
    ) -> MemoryRecord:
        clean = _clean(content)
        if not clean:
            raise ValueError("memory content must not be empty")
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        now = _now()
        digest = _digest(clean)
        tag_values = tuple(sorted({_clean(tag) for tag in tags if _clean(tag)}))
        existing = self.connection.execute(
            """
            SELECT * FROM memories
            WHERE session_id = ? AND kind = ? AND content_hash = ?
            """,
            (session_id, kind.value, digest),
        ).fetchone()
        if existing is not None:
            memory_id = int(existing["id"])
            merged_tags = tuple(
                sorted(set(json.loads(str(existing["tags_json"]))) | set(tag_values))
            )
            self.connection.execute(
                """
                UPDATE memories
                SET state = ?, content = ?, trace_hint = '', source = ?,
                    destination = ?, tags_json = ?, salience = ?,
                    confidence = ?, arousal = ?,
                    repetition_count = repetition_count + 1,
                    protected = ?, updated_at = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (
                    MemoryState.ACTIVE.value,
                    clean,
                    source,
                    destination,
                    json.dumps(merged_tags),
                    max(float(existing["salience"]), _clamp(salience)),
                    max(float(existing["confidence"]), _clamp(confidence)),
                    max(float(existing["arousal"]), _clamp(arousal)),
                    int(bool(existing["protected"]) or protected),
                    _iso(now),
                    _iso(now),
                    memory_id,
                ),
            )
        else:
            cursor = self.connection.execute(
                """
                INSERT INTO memories(
                    session_id, kind, state, evidence_status, content,
                    trace_hint, content_hash, source, destination, tags_json,
                    salience, confidence, arousal, repetition_count,
                    retrieval_count, protected, created_at, updated_at,
                    last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind.value,
                    MemoryState.ACTIVE.value,
                    evidence_status.value,
                    clean,
                    digest,
                    source,
                    destination,
                    json.dumps(tag_values),
                    _clamp(salience),
                    _clamp(confidence),
                    _clamp(arousal),
                    int(protected),
                    _iso(now),
                    _iso(now),
                    _iso(now),
                ),
            )
            memory_id = int(cursor.lastrowid)
        self.connection.commit()
        return self.get(memory_id)

    def recall(
        self,
        query: str,
        *,
        session_id: str = "default",
        record_retrieval: bool = True,
    ) -> tuple[RecallResult, ...]:
        clean_query = _clean(query)
        if not clean_query:
            return ()
        rows = self.connection.execute(
            """
            SELECT * FROM memories
            WHERE session_id IN (?, 'global')
              AND state IN (?, ?, ?)
              AND content <> ''
            """,
            (
                session_id,
                MemoryState.ACTIVE.value,
                MemoryState.CONSOLIDATED.value,
                MemoryState.LATENT.value,
            ),
        ).fetchall()
        results: list[RecallResult] = []
        for row in rows:
            memory = self._row(row)
            relevance = lexical_relevance(clean_query, memory)
            if relevance <= 0.0:
                continue
            retention = retention_score(memory, policy=self.policy)
            evidence_bonus = {
                EvidenceStatus.VERIFIED: 0.12,
                EvidenceStatus.EVIDENCE_SUPPORTED: 0.08,
                EvidenceStatus.USER_SUPPLIED: 0.03,
                EvidenceStatus.PROVISIONAL: 0.0,
                EvidenceStatus.MODEL_GENERATED: -0.05,
            }[memory.evidence_status]
            score = _clamp(
                0.60 * relevance
                + 0.35 * retention
                + evidence_bonus
                + 0.05 * memory.salience
            )
            if score >= 0.10:
                results.append(RecallResult(memory, relevance, retention, score))
        results.sort(key=lambda item: (-item.score, item.memory.id))

        selected: list[RecallResult] = []
        used = 0
        for result in results:
            rendered = self._render(result)
            if used + len(rendered) + 1 > self.policy.prompt_budget_chars:
                continue
            selected.append(result)
            used += len(rendered) + 1
            if len(selected) >= self.policy.max_recall_items:
                break

        if record_retrieval and selected:
            ids = [item.memory.id for item in selected]
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(
                f"""
                UPDATE memories
                SET retrieval_count = retrieval_count + 1,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (_iso(_now()), *ids),
            )
            self.connection.commit()
        return tuple(selected)

    @staticmethod
    def _render(result: RecallResult) -> str:
        memory = result.memory
        return (
            f"[memory id={memory.id} kind={memory.kind.value} "
            f"state={memory.state.value} evidence={memory.evidence_status.value} "
            f"score={result.score:.3f} source={memory.source}] {memory.content}"
        )

    def render_context(self, query: str, *, session_id: str = "default") -> str:
        return "\n".join(
            self._render(result)
            for result in self.recall(query, session_id=session_id)
        )

    def maintain(
        self,
        *,
        now: Optional[datetime] = None,
        apply: bool = False,
    ) -> MaintenanceReport:
        current = now or _now()
        rows = self.connection.execute(
            """
            SELECT * FROM memories
            WHERE state IN (?, ?, ?)
            """,
            (
                MemoryState.ACTIVE.value,
                MemoryState.CONSOLIDATED.value,
                MemoryState.LATENT.value,
            ),
        ).fetchall()
        actions: list[MaintenanceAction] = []
        for row in rows:
            memory = self._row(row)
            retention = retention_score(memory, now=current, policy=self.policy)
            age_hours = max(
                0.0,
                (current - memory.last_seen_at).total_seconds() / 3600.0,
            )
            next_state = MemoryState.ACTIVE
            reason = "available"
            if memory.protected or memory.kind is MemoryKind.CORE:
                next_state = (
                    MemoryState.CONSOLIDATED
                    if retention >= self.policy.consolidate_threshold
                    else MemoryState.ACTIVE
                )
                reason = "protected_from_pruning"
            elif (
                retention < self.policy.trace_threshold
                and age_hours >= self.policy.trace_min_age_hours
            ):
                next_state = MemoryState.TRACE
                reason = "weak_old_memory_compressed_to_trace"
            elif retention < self.policy.latent_threshold:
                next_state = MemoryState.LATENT
                reason = "low_retrieval_strength"
            elif (
                retention >= self.policy.consolidate_threshold
                and memory.repetition_count + memory.retrieval_count >= 3
            ):
                next_state = MemoryState.CONSOLIDATED
                reason = "retrieval_or_repetition_consolidation"
            if next_state is memory.state:
                continue
            action = MaintenanceAction(
                memory.id,
                memory.state,
                next_state,
                retention,
                reason,
            )
            actions.append(action)
            if apply:
                content = memory.content
                trace_hint = memory.trace_hint
                kind = memory.kind
                if next_state is MemoryState.TRACE:
                    trace_hint = _make_trace(
                        memory.content,
                        memory.tags,
                        memory.destination,
                    )
                    content = ""
                    kind = MemoryKind.TRACE
                self.connection.execute(
                    """
                    UPDATE memories
                    SET kind = ?, state = ?, content = ?, trace_hint = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        kind.value,
                        next_state.value,
                        content,
                        trace_hint,
                        _iso(current),
                        memory.id,
                    ),
                )
        if apply:
            self.connection.commit()
        return MaintenanceReport(apply, len(rows), tuple(actions))

    def maintain_if_due(self) -> Optional[MaintenanceReport]:
        row = self.connection.execute(
            "SELECT value FROM memory_meta WHERE key = 'interaction_count'"
        ).fetchone()
        count = int(row["value"]) + 1 if row else 1
        self.connection.execute(
            """
            INSERT INTO memory_meta(key, value) VALUES('interaction_count', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(count),),
        )
        self.connection.commit()
        if count % max(1, self.policy.maintenance_interval):
            return None
        return self.maintain(apply=True)

    def forget(self, memory_id: int, *, confirmation: str) -> MemoryRecord:
        expected = f"FORGET-{memory_id}"
        if confirmation != expected:
            raise PermissionError(f"Expected explicit confirmation {expected!r}")
        self.connection.execute(
            """
            UPDATE memories
            SET state = ?, content = '', trace_hint = '', updated_at = ?
            WHERE id = ?
            """,
            (MemoryState.FORGOTTEN.value, _iso(_now()), memory_id),
        )
        self.connection.commit()
        return self.get(memory_id)

    def status(self) -> dict[str, object]:
        state_rows = self.connection.execute(
            "SELECT state, COUNT(*) AS count FROM memories GROUP BY state"
        ).fetchall()
        kind_rows = self.connection.execute(
            "SELECT kind, COUNT(*) AS count FROM memories GROUP BY kind"
        ).fetchall()
        return {
            "database": str(self.db_path),
            "states": {str(row["state"]): int(row["count"]) for row in state_rows},
            "kinds": {str(row["kind"]): int(row["count"]) for row in kind_rows},
            "policy": asdict(self.policy),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garvis-memory")
    sub = parser.add_subparsers(dest="command", required=True)

    remember = sub.add_parser("remember")
    remember.add_argument("content")
    remember.add_argument(
        "--kind",
        choices=[item.value for item in MemoryKind],
        default=MemoryKind.EPISODIC.value,
    )
    remember.add_argument(
        "--evidence",
        choices=[item.value for item in EvidenceStatus],
        default=EvidenceStatus.USER_SUPPLIED.value,
    )
    remember.add_argument("--protected", action="store_true")

    recall = sub.add_parser("recall")
    recall.add_argument("query")

    maintain = sub.add_parser("maintain")
    maintain.add_argument("--apply", action="store_true")

    sub.add_parser("status")

    forget = sub.add_parser("forget")
    forget.add_argument("memory_id", type=int)
    forget.add_argument("--confirm", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with MemoryStore.from_environment() as store:
            if args.command == "remember":
                record = store.remember(
                    args.content,
                    kind=MemoryKind(args.kind),
                    evidence_status=EvidenceStatus(args.evidence),
                    protected=args.protected,
                )
                print(json.dumps({"memory_id": record.id, "state": record.state.value}))
            elif args.command == "recall":
                print(store.render_context(args.query))
            elif args.command == "maintain":
                report = store.maintain(apply=args.apply)
                print(
                    json.dumps(
                        {
                            "applied": report.applied,
                            "scanned": report.scanned,
                            "actions": [
                                {
                                    "memory_id": action.memory_id,
                                    "from": action.previous_state.value,
                                    "to": action.next_state.value,
                                    "retention": round(action.retention, 4),
                                    "reason": action.reason,
                                }
                                for action in report.actions
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.command == "status":
                print(json.dumps(store.status(), indent=2, sort_keys=True))
            elif args.command == "forget":
                record = store.forget(args.memory_id, confirmation=args.confirm)
                print(json.dumps({"memory_id": record.id, "state": record.state.value}))
    except Exception as exc:
        print(f"GARVIS memory error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
