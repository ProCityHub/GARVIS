"""Durable autonomous upgrade-cycle state machine for GARVIS THANOS MODE.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

A cycle is a persistent record of one self-maintenance attempt. Every state
transition is appended to durable storage before it takes effect, so a cycle
interrupted by Android suspension, network loss, or process termination
resumes from its last recorded state rather than restarting.

Only one cycle may hold the modification lock at a time.

Python 3.9 compatible. Termux-safe: no fcntl, no daemon threads, atomic
``os.replace`` writes, and a stale-lock reaper based on process liveness.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from garvis.stage_gate import canonical_json, new_identifier, utc_now_iso
from garvis.thanos_mode import (
    MergeDecision,
    ThanosAction,
    ThanosAuthorization,
    ThanosError,
    permits,
)

__all__ = [
    "CycleLock",
    "CycleLockError",
    "CycleState",
    "CycleStore",
    "CycleTransitionError",
    "TERMINAL_STATES",
    "UpgradeCycle",
    "advance",
    "record_merge_decision",
    "transition_is_legal",
]


class CycleState(str, Enum):
    """States of one autonomous upgrade cycle."""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    RESEARCHING = "RESEARCHING"
    DIAGNOSING = "DIAGNOSING"
    SPECIFYING = "SPECIFYING"
    PLANNING = "PLANNING"
    PREPARING_WORKSPACE = "PREPARING_WORKSPACE"
    PATCHING = "PATCHING"
    FORMATTING = "FORMATTING"
    LINTING = "LINTING"
    TYPE_CHECKING = "TYPE_CHECKING"
    TESTING = "TESTING"
    SECURITY_REVIEWING = "SECURITY_REVIEWING"
    PACKAGING = "PACKAGING"
    COMMITTING = "COMMITTING"
    PUSHING = "PUSHING"
    OPENING_PR = "OPENING_PR"
    WAITING_FOR_CI = "WAITING_FOR_CI"
    REPAIRING_CI = "REPAIRING_CI"
    MERGING = "MERGING"
    SYNCHRONIZING = "SYNCHRONIZING"
    RESTARTING = "RESTARTING"
    HEALTH_CHECKING = "HEALTH_CHECKING"
    PROMOTING = "PROMOTING"
    ROLLING_BACK = "ROLLING_BACK"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    REVOKED = "REVOKED"


TERMINAL_STATES = frozenset(
    {
        CycleState.COMPLETED,
        CycleState.FAILED,
        CycleState.REVOKED,
    }
)

#: States a cycle may enter from anywhere: owner control and hard stops.
_UNIVERSAL = frozenset(
    {
        CycleState.PAUSED,
        CycleState.REVOKED,
        CycleState.BLOCKED,
        CycleState.FAILED,
    }
)

_PIPELINE = (
    CycleState.IDLE,
    CycleState.OBSERVING,
    CycleState.RESEARCHING,
    CycleState.DIAGNOSING,
    CycleState.SPECIFYING,
    CycleState.PLANNING,
    CycleState.PREPARING_WORKSPACE,
    CycleState.PATCHING,
    CycleState.FORMATTING,
    CycleState.LINTING,
    CycleState.TYPE_CHECKING,
    CycleState.TESTING,
    CycleState.SECURITY_REVIEWING,
    CycleState.PACKAGING,
    CycleState.COMMITTING,
    CycleState.PUSHING,
    CycleState.OPENING_PR,
    CycleState.WAITING_FOR_CI,
    CycleState.MERGING,
    CycleState.SYNCHRONIZING,
    CycleState.RESTARTING,
    CycleState.HEALTH_CHECKING,
    CycleState.PROMOTING,
    CycleState.COMPLETED,
)


def _build_transitions() -> dict:
    table: dict = {}
    for index, state in enumerate(_PIPELINE):
        allowed = set(_UNIVERSAL)
        if index + 1 < len(_PIPELINE):
            allowed.add(_PIPELINE[index + 1])
        table[state] = allowed

    # Validation failures fall back to PATCHING for a bounded repair.
    for state in (
        CycleState.FORMATTING,
        CycleState.LINTING,
        CycleState.TYPE_CHECKING,
        CycleState.TESTING,
        CycleState.SECURITY_REVIEWING,
        CycleState.PACKAGING,
    ):
        table[state].add(CycleState.PATCHING)

    table[CycleState.WAITING_FOR_CI].add(CycleState.REPAIRING_CI)
    table[CycleState.REPAIRING_CI] = set(_UNIVERSAL) | {
        CycleState.PATCHING,
        CycleState.PUSHING,
        CycleState.WAITING_FOR_CI,
    }
    table[CycleState.HEALTH_CHECKING].add(CycleState.ROLLING_BACK)
    table[CycleState.PROMOTING].add(CycleState.ROLLING_BACK)
    table[CycleState.ROLLING_BACK] = set(_UNIVERSAL) | {
        CycleState.COMPLETED,
        CycleState.OBSERVING,
    }
    table[CycleState.PAUSED] = set(_UNIVERSAL) | set(_PIPELINE)
    table[CycleState.BLOCKED] = set(_UNIVERSAL) | {CycleState.OBSERVING}
    for terminal in TERMINAL_STATES:
        table[terminal] = set()
    return table


_TRANSITIONS = _build_transitions()


class CycleTransitionError(RuntimeError):
    """Raised on an illegal cycle-state transition."""


def transition_is_legal(source: CycleState, destination: CycleState) -> bool:
    """Return True when ``source -> destination`` is a permitted move."""

    return destination in _TRANSITIONS.get(source, set())


@dataclass(frozen=True)
class CycleTransition:
    """One recorded state change."""

    source: str
    destination: str
    occurred_at: str
    note: str = ""

    def to_payload(self) -> dict:
        return {
            "source": self.source,
            "destination": self.destination,
            "occurred_at": self.occurred_at,
            "note": self.note,
        }


@dataclass(frozen=True)
class UpgradeCycle:
    """Durable record of one autonomous upgrade attempt."""

    cycle_id: str
    objective: str
    starting_commit: str
    starting_version: str
    state: CycleState = CycleState.IDLE
    created_at: str = ""
    updated_at: str = ""
    transitions: tuple[CycleTransition, ...] = ()
    changed_files: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)
    pull_request_number: int | None = None
    commit_sha: str | None = None
    merge_sha: str | None = None
    last_known_good: str | None = None
    blocker: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_payload(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "objective": self.objective,
            "starting_commit": self.starting_commit,
            "starting_version": self.starting_version,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transitions": [t.to_payload() for t in self.transitions],
            "changed_files": list(self.changed_files),
            "evidence": dict(self.evidence),
            "pull_request_number": self.pull_request_number,
            "commit_sha": self.commit_sha,
            "merge_sha": self.merge_sha,
            "last_known_good": self.last_known_good,
            "blocker": self.blocker,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> UpgradeCycle:
        return cls(
            cycle_id=str(payload["cycle_id"]),
            objective=str(payload["objective"]),
            starting_commit=str(payload["starting_commit"]),
            starting_version=str(payload["starting_version"]),
            state=CycleState(payload["state"]),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            transitions=tuple(
                CycleTransition(
                    source=str(t["source"]),
                    destination=str(t["destination"]),
                    occurred_at=str(t["occurred_at"]),
                    note=str(t.get("note", "")),
                )
                for t in payload.get("transitions", [])
            ),
            changed_files=tuple(str(f) for f in payload.get("changed_files", [])),
            evidence=dict(payload.get("evidence", {})),
            pull_request_number=payload.get("pull_request_number"),
            commit_sha=payload.get("commit_sha"),
            merge_sha=payload.get("merge_sha"),
            last_known_good=payload.get("last_known_good"),
            blocker=payload.get("blocker"),
        )


def new_cycle(
    *,
    objective: str,
    starting_commit: str,
    starting_version: str,
    last_known_good: str | None = None,
    now: str | None = None,
) -> UpgradeCycle:
    """Create a fresh cycle in IDLE."""

    timestamp = now or utc_now_iso()
    return UpgradeCycle(
        cycle_id=new_identifier("cycle"),
        objective=objective,
        starting_commit=starting_commit,
        starting_version=starting_version,
        created_at=timestamp,
        updated_at=timestamp,
        last_known_good=last_known_good or starting_commit,
    )


def advance(
    cycle: UpgradeCycle,
    destination: CycleState,
    authorization: ThanosAuthorization,
    *,
    action: ThanosAction = ThanosAction.CONTINUE_UPGRADING,
    note: str = "",
    now: str | None = None,
    **updates: Any,
) -> UpgradeCycle:
    """Return ``cycle`` moved to ``destination``, recording the transition.

    The standing authorization is re-verified on every transition. It is
    never consumed: no per-stage prompt is emitted.
    """

    if cycle.is_terminal:
        raise CycleTransitionError(f"cycle {cycle.cycle_id} is terminal in {cycle.state.value}")
    if not transition_is_legal(cycle.state, destination):
        raise CycleTransitionError(f"illegal transition {cycle.state.value} -> {destination.value}")

    if destination not in (CycleState.PAUSED, CycleState.REVOKED):
        try:
            permits(authorization, action)
        except ThanosError as error:
            timestamp = now or utc_now_iso()
            return replace(
                cycle,
                state=CycleState.BLOCKED,
                blocker=str(error),
                updated_at=timestamp,
                transitions=cycle.transitions
                + (
                    CycleTransition(
                        source=cycle.state.value,
                        destination=CycleState.BLOCKED.value,
                        occurred_at=timestamp,
                        note=str(error),
                    ),
                ),
            )

    timestamp = now or utc_now_iso()
    return replace(
        cycle,
        state=destination,
        updated_at=timestamp,
        transitions=cycle.transitions
        + (
            CycleTransition(
                source=cycle.state.value,
                destination=destination.value,
                occurred_at=timestamp,
                note=note,
            ),
        ),
        **updates,
    )


def record_merge_decision(
    cycle: UpgradeCycle,
    decision: MergeDecision,
    authorization: ThanosAuthorization,
    *,
    now: str | None = None,
) -> UpgradeCycle:
    """Apply a merge-gate decision to a cycle waiting in MERGING."""

    if decision.allowed:
        return advance(
            cycle,
            CycleState.SYNCHRONIZING,
            authorization,
            action=ThanosAction.MERGE,
            note="autonomous squash merge: all preconditions satisfied",
            now=now,
        )
    return advance(
        cycle,
        CycleState.BLOCKED,
        authorization,
        action=ThanosAction.MONITOR_CI,
        note="; ".join(decision.blocking_reasons),
        now=now,
        blocker="; ".join(decision.blocking_reasons),
    )


class CycleStore:
    """Atomic durable persistence for the active and historical cycles."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict:
        if not self._path.is_file():
            return {"active": None, "history": []}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"active": None, "history": [], "corrupt": True}

    def is_corrupt(self) -> bool:
        return bool(self._read().get("corrupt"))

    def save(self, cycle: UpgradeCycle) -> UpgradeCycle:
        """Persist ``cycle`` as the active cycle, archiving it when terminal."""

        data = self._read()
        data.pop("corrupt", None)
        history = list(data.get("history") or [])
        if cycle.is_terminal:
            history.append(cycle.to_payload())
            data["active"] = None
        else:
            data["active"] = cycle.to_payload()
        data["history"] = history[-50:]
        self._atomic_write(data)
        return cycle

    def resume(self) -> UpgradeCycle | None:
        """Return the interrupted active cycle, or None."""

        payload = self._read().get("active")
        if not payload:
            return None
        return UpgradeCycle.from_payload(payload)

    def history(self) -> tuple[UpgradeCycle, ...]:
        return tuple(UpgradeCycle.from_payload(p) for p in self._read().get("history") or [])

    def _atomic_write(self, payload: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".cycle-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self._path))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise


class CycleLockError(RuntimeError):
    """Raised when the modification lock cannot be acquired."""


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class CycleLock:
    """Single-writer lock. Termux-safe: O_EXCL file, no fcntl, PID reaping."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> CycleLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._reap_stale():
                return self.acquire()
            raise CycleLockError(f"another upgrade cycle holds {self._path}") from None
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "acquired_at": utc_now_iso()}, stream)
        self._held = True
        return self

    def _reap_stale(self) -> bool:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
        except (OSError, ValueError, TypeError):
            pid = -1
        if _process_alive(pid):
            return False
        try:
            os.unlink(str(self._path))
        except OSError:
            return False
        return True

    def release(self) -> None:
        if self._held:
            try:
                os.unlink(str(self._path))
            except OSError:
                pass
            self._held = False

    def __enter__(self) -> CycleLock:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


# Keep the import referenced for Termux signal-safety documentation purposes.
_SIGNALS_OF_INTEREST = (signal.SIGTERM, signal.SIGINT)
