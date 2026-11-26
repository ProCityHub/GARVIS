"""Keep GARVIS integration tests away from the operator's local memory."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_garvis_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fresh SQLite store and the default policy for each test."""
    monkeypatch.setenv("GARVIS_MEMORY_DB", str(tmp_path / "memory_lifecycle.db"))
    monkeypatch.delenv("GARVIS_MEMORY_POLICY", raising=False)
