from pathlib import Path

from garvis.economics import LedgerEvent, OpportunityLedger


def test_ledger_is_append_only_and_totals_events(tmp_path: Path) -> None:
    ledger = OpportunityLedger(tmp_path / "ledger.jsonl")
    ledger.append(LedgerEvent("job-1", "revenue", 500.0, "deposit"))
    ledger.append(LedgerEvent("job-1", "cost", 125.0, "materials"))

    assert [event.event_type for event in ledger.events()] == ["revenue", "cost"]
    assert ledger.total_for("revenue") == 500.0
    assert ledger.total_for("cost") == 125.0
