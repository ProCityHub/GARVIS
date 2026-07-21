"""Append-only local ledger for opportunity outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class LedgerEvent:
    opportunity_id: str
    event_type: str
    amount: float = 0.0
    note: str = ""
    occurred_at: str = ""

    def __post_init__(self) -> None:
        if not self.opportunity_id.strip():
            raise ValueError("opportunity_id must not be empty")
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")

    def normalized(self) -> "LedgerEvent":
        timestamp = self.occurred_at or datetime.now(timezone.utc).isoformat()
        return LedgerEvent(
            opportunity_id=self.opportunity_id,
            event_type=self.event_type,
            amount=float(self.amount),
            note=self.note,
            occurred_at=timestamp,
        )


class OpportunityLedger:
    """Store immutable JSONL events without bank or payment access."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: LedgerEvent) -> LedgerEvent:
        normalized = event.normalized()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(normalized), sort_keys=True) + "\n")
        return normalized

    def events(self) -> Iterator[LedgerEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield LedgerEvent(**payload)

    def total_for(self, event_type: str) -> float:
        return sum(event.amount for event in self.events() if event.event_type == event_type)
