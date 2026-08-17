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

    def test_research_memory_always_surfaces_protected_core(self) -> None:
        self.store.remember(
            "Protected provenance must remain visible during research",
            session_id="global",
            kind=MemoryKind.CORE,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            source="research_memory_test",
            protected=True,
        )

        context = self.store.render_research_context(
            "a completely unrelated experiment"
        )

        self.assertIn("consulted=true", context)
        self.assertIn("prominence=required", context)
        self.assertIn("execution_authority=false", context)
        self.assertIn(
            "Protected provenance must remain visible during research",
            context,
        )

    def test_prospective_memory_is_first_class_and_recallable(self) -> None:
        self.store.remember(
            "When contradiction appears return control to the foreground",
            kind=MemoryKind.PROSPECTIVE,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            tags=("contradiction", "foreground", "intention"),
        )

        context = self.store.render_research_context(
            "contradiction foreground"
        )

        self.assertIn("kind=prospective", context)
        self.assertIn(
            "return control to the foreground",
            context,
        )

    def test_simulation_memory_cannot_claim_verified_evidence(self) -> None:
        with self.assertRaises(ValueError):
            self.store.remember(
                "Imagined future result",
                kind=MemoryKind.SIMULATION,
                evidence_status=EvidenceStatus.VERIFIED,
            )

        simulation = self.store.remember(
            "Imagined future result",
            kind=MemoryKind.SIMULATION,
            evidence_status=EvidenceStatus.MODEL_GENERATED,
        )

        self.assertEqual(
            simulation.evidence_status,
            EvidenceStatus.MODEL_GENERATED,
        )

        context = self.store.render_research_context(
            "imagined future result"
        )

        self.assertIn("simulation_is_evidence=false", context)
        self.assertIn("kind=simulation", context)
        self.assertIn(
            "evidence=model_generated_unverified",
            context,
        )

    def test_procedural_memory_can_be_background_candidate(self) -> None:
        record = self.store.remember(
            "For build verification inspect test output and status",
            kind=MemoryKind.PROCEDURAL,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            tags=("build", "verification", "procedure"),
        )

        before = self.store.get(record.id)

        signal = self.store.automatic_memory_control(
            "build verification test output"
        )

        after = self.store.get(record.id)

        self.assertTrue(signal.procedural_candidates)
        self.assertFalse(signal.prospective_triggers)
        self.assertFalse(signal.foreground_required)
        self.assertFalse(signal.execution_authority)
        self.assertFalse(signal.silent_consolidation_allowed)
        self.assertEqual(
            signal.reason,
            "procedural_candidate_available",
        )
        self.assertEqual(
            before.retrieval_count,
            after.retrieval_count,
        )

    def test_prospective_cue_requires_foreground_reengagement(self) -> None:
        self.store.remember(
            "When the merge head changes return to explicit review",
            kind=MemoryKind.PROSPECTIVE,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            tags=("merge", "head", "review"),
        )

        signal = self.store.automatic_memory_control(
            "merge head changes"
        )

        self.assertTrue(signal.prospective_triggers)
        self.assertTrue(signal.foreground_required)
        self.assertFalse(signal.execution_authority)
        self.assertEqual(
            signal.reason,
            "prospective_cue",
        )

    def test_contradiction_forces_foreground_reengagement(self) -> None:
        signal = self.store.automatic_memory_control(
            "unrelated observation",
            contradiction_observed=True,
        )

        self.assertTrue(signal.contradiction_observed)
        self.assertTrue(signal.foreground_required)
        self.assertFalse(signal.execution_authority)
        self.assertFalse(signal.silent_consolidation_allowed)
        self.assertEqual(signal.reason, "contradiction")

    def test_contradiction_does_not_silently_strengthen_memory(self) -> None:
        record = self.store.remember(
            "Routine procedure remains bounded by observation",
            kind=MemoryKind.PROCEDURAL,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            tags=("routine", "observation"),
        )

        before = self.store.get(record.id)

        signal = self.store.automatic_memory_control(
            "routine observation",
            contradiction_observed=True,
        )

        after = self.store.get(record.id)

        self.assertTrue(signal.foreground_required)
        self.assertFalse(signal.silent_consolidation_allowed)
        self.assertEqual(
            before.retrieval_count,
            after.retrieval_count,
        )
        self.assertEqual(
            before.repetition_count,
            after.repetition_count,
        )
        self.assertEqual(before.state, after.state)
        self.assertEqual(before.content, after.content)

    def test_research_context_does_not_silently_strengthen_retrieval(self) -> None:
        record = self.store.remember(
            "Research memory consultation must remain observational",
            kind=MemoryKind.SEMANTIC,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            tags=("research", "memory", "observation"),
        )

        before = self.store.get(record.id)

        context = self.store.render_research_context(
            "research memory observation"
        )

        after = self.store.get(record.id)

        self.assertIn(
            "Research memory consultation must remain observational",
            context,
        )
        self.assertEqual(
            before.retrieval_count,
            after.retrieval_count,
        )
        self.assertEqual(
            before.repetition_count,
            after.repetition_count,
        )
        self.assertEqual(before.state, after.state)

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



def test_research_memory_classifier_covers_adversarial_research_language() -> None:
    from garvis.memory_lifecycle import research_memory_required

    positives = (
        "Research current Python release changes",
        "Browse the web for current Python release changes",
        "Search the internet for current drywall prices",
        "Look up the latest Python documentation online",
        "Check today's weather in Philadelphia",
        "What is today's weather in Philadelphia?",
        "What are current drywall prices?",
        "Find recent primary evidence about memory consolidation",
        "Compare the latest studies on procedural memory",
        "Investigate current evidence for this hypothesis",
        "Find out whether Python 3.14 has been released",
        "What changed recently in OpenAI API documentation?",
        "Research procedural memory architecture",
    )

    negatives = (
        "Explain how drywall compound cures",
        "Summarize this local file",
        "Calculate 1/phi + 1/phi^2",
    )

    for message in positives:
        assert research_memory_required(message), message

    for message in negatives:
        assert not research_memory_required(message), message
