from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from garvis.memory_lifecycle import (
    EvidenceStatus,
    MemoryKind,
    MemoryPolicy,
    MemoryState,
    MemoryStore,
    retention_score,
)


class MemoryLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_duplicate_reinforces_instead_of_duplicating(self) -> None:
        first = self.store.remember("Use a local GGUF runtime")
        second = self.store.remember("  Use a local GGUF runtime  ")
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.repetition_count, 2)

    def test_recall_is_relevant_bounded_and_evidence_labeled(self) -> None:
        self.store.remember(
            "GARVIS uses local GGUF model weights",
            kind=MemoryKind.SEMANTIC,
            evidence_status=EvidenceStatus.EVIDENCE_SUPPORTED,
            salience=0.9,
            confidence=0.9,
        )
        self.store.remember("Drywall estimate for a bedroom")
        context = self.store.render_context("local model GGUF")
        self.assertIn("local GGUF model weights", context)
        self.assertIn("evidence=evidence_supported", context)
        self.assertNotIn("Drywall estimate", context)
        self.assertLessEqual(len(context), self.store.policy.prompt_budget_chars)

    def test_retention_fades(self) -> None:
        record = self.store.remember(
            "Temporary working memory",
            kind=MemoryKind.WORKING,
        )
        early = retention_score(record, now=record.last_seen_at)
        late = retention_score(
            record,
            now=record.last_seen_at + timedelta(days=7),
        )
        self.assertGreater(early, late)

    def test_old_weak_memory_becomes_nonfactual_trace(self) -> None:
        policy = MemoryPolicy(
            working_half_life_hours=0.1,
            trace_threshold=0.9,
            latent_threshold=0.95,
            trace_min_age_hours=0.0,
        )
        self.store.close()
        self.store = MemoryStore(Path(self.temp.name) / "trace.db", policy)
        record = self.store.remember(
            "Disposable wording should be pruned but leave a topic trace",
            kind=MemoryKind.WORKING,
            tags=("temporary",),
            salience=0.0,
            confidence=0.0,
        )
        self.store.maintain(
            now=record.last_seen_at + timedelta(days=2),
            apply=True,
        )
        updated = self.store.get(record.id)
        self.assertEqual(updated.state, MemoryState.TRACE)
        self.assertEqual(updated.kind, MemoryKind.TRACE)
        self.assertEqual(updated.content, "")
        self.assertIn("keywords=", updated.trace_hint)

    def test_protected_core_memory_is_not_auto_traced(self) -> None:
        policy = MemoryPolicy(
            core_half_life_hours=0.01,
            trace_threshold=0.99,
            latent_threshold=0.999,
            trace_min_age_hours=0.0,
        )
        self.store.close()
        self.store = MemoryStore(Path(self.temp.name) / "core.db", policy)
        record = self.store.remember(
            "Adrien protected this boundary",
            kind=MemoryKind.CORE,
            protected=True,
        )
        self.store.maintain(
            now=record.last_seen_at + timedelta(days=500),
            apply=True,
        )
        updated = self.store.get(record.id)
        self.assertNotEqual(updated.state, MemoryState.TRACE)
        self.assertTrue(updated.content)

    def test_explicit_forgetting_requires_token(self) -> None:
        record = self.store.remember("Only forget with confirmation")
        with self.assertRaises(PermissionError):
            self.store.forget(record.id, confirmation="yes")
        forgotten = self.store.forget(
            record.id,
            confirmation=f"FORGET-{record.id}",
        )
        self.assertEqual(forgotten.state, MemoryState.FORGOTTEN)
        self.assertEqual(forgotten.content, "")


if __name__ == "__main__":
    unittest.main()
