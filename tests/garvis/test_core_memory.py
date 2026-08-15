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


    def test_attribution_notice_rejects_tampered_manifest(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from garvis.core_memory import attribution_notice, load_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            data = load_manifest()
            data["attribution"] = (
                str(data["attribution"]) + " [tampered]"
            )
            path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                attribution_notice(path)


    def test_tampered_manifest_cannot_enter_protected_core_memory(
        self,
    ) -> None:
        import json
        import tempfile
        from pathlib import Path

        from garvis.core_memory import ensure_core_memories, load_manifest
        from garvis.memory_lifecycle import MemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            data = load_manifest()
            data["attribution"] = (
                str(data["attribution"]) + " [tampered]"
            )
            path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with MemoryStore(Path(tmp) / "memory.db") as store:
                with self.assertRaises(ValueError):
                    ensure_core_memories(store, path)

                count = store.connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    WHERE source = 'garvis_core_memory_manifest'
                    """
                ).fetchone()[0]

                self.assertEqual(int(count), 0)


    def test_verified_manifest_preserves_core_memory_behavior(
        self,
    ) -> None:
        import tempfile
        from pathlib import Path

        from garvis.core_memory import (
            ensure_core_memories,
            render_core_context,
        )
        from garvis.memory_lifecycle import MemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            with MemoryStore(Path(tmp) / "memory.db") as store:
                first = ensure_core_memories(store)
                second = ensure_core_memories(store)

                self.assertEqual(first, second)
                self.assertIn(
                    "Adrien D. Thomas",
                    render_core_context(store),
                )


if __name__ == "__main__":
    unittest.main()
