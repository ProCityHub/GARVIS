from pathlib import Path
import tempfile
import unittest

from garvis.neurocognitive import NeurocognitiveEngine
from garvis.neurocognitive.models import EvidenceStatus, MemoryKind


class NeurocognitiveEngineTests(unittest.TestCase):
    def test_cycle_is_bounded_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = NeurocognitiveEngine(
                db_path=Path(directory) / "memory.db",
                context_budget_chars=4_000,
            )
            cycle = engine.prepare(
                "I want GARVIS to remember the hypercube heartbeat.",
                session_id="test",
            )
            self.assertIn("0.0", cycle.model_context)
            self.assertLessEqual(len(cycle.model_context), 4_000)

            engine.consolidate(
                cycle=cycle,
                assistant_text="I will use selective recall.",
            )
            second = engine.prepare("What heartbeat do I want?", session_id="test")
            self.assertTrue(second.recalled)

    def test_dream_questions_are_speculative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = NeurocognitiveEngine(db_path=Path(directory) / "memory.db")
            cycle = engine.prepare(
                "Could the heartbeat model context like human memory?",
                session_id="test",
            )
            engine.consolidate(cycle=cycle, assistant_text="That is a hypothesis.")
            memories = engine.store.list_memories(session_id="test")
            dream = next(memory for memory in memories if memory.kind is MemoryKind.DREAM)
            self.assertIs(dream.status, EvidenceStatus.SPECULATIVE)


if __name__ == "__main__":
    unittest.main()
