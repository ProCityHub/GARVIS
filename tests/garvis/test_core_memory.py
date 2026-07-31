from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from garvis.core_memory import (
    DEFAULT_MANIFEST,
    attribution_notice,
    ensure_core_memories,
    export_agent_bootstrap,
    render_core_context,
    verify_manifest,
)
from garvis.memory_lifecycle import MemoryKind, MemoryStore


class CoreMemoryTests(unittest.TestCase):
    def test_manifest(self) -> None:
        status = verify_manifest()
        self.assertTrue(status.compatible, status.reason)
        self.assertEqual(status.creator, "Adrien D. Thomas")
        self.assertIn("Adrien D. Thomas", attribution_notice())

    def test_global_core_memory_cross_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryStore(Path(tmp) / "memory.db") as store:
                first = ensure_core_memories(store)
                second = ensure_core_memories(store)
                self.assertEqual(first, second)
                self.assertIn("Adrien D. Thomas", render_core_context(store))
                recalled = store.render_context("Who created GARVIS?", session_id="new-chat")
                self.assertIn("Adrien D. Thomas", recalled)
                row = store.connection.execute(
                    "SELECT kind, protected FROM memories WHERE session_id='global' "
                    "AND destination='identity_provenance'"
                ).fetchone()
                self.assertEqual(row["kind"], MemoryKind.CORE.value)
                self.assertEqual(int(row["protected"]), 1)

    def test_agent_export(self) -> None:
        adapter = export_agent_bootstrap()
        self.assertTrue(adapter["official_compatible"])
        self.assertIn("Adrien D. Thomas", str(adapter["instructions"]))

    def test_tampering_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            data = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
            data["creator"] = "Removed"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertFalse(verify_manifest(path).compatible)


if __name__ == "__main__":
    unittest.main()
