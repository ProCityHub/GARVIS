from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import unittest

from garvis.assistant import GarvisAssistant
from garvis.neurocognitive.consolidation import consolidate_turn
from garvis.neurocognitive.models import EvidenceStatus, MemoryKind
from garvis.neurocognitive.recall import recall
from garvis.neurocognitive.store import NeuroStore


class FakeResult:
    final_output = "direct answer"


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, agent: Any, input_text: str, **kwargs: Any) -> FakeResult:
        self.calls.append({"input": input_text, **kwargs})
        return FakeResult()


class DialogueFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_neuro_chat_can_disable_repository_grounding(self) -> None:
        runner = FakeRunner()
        assistant = GarvisAssistant(runner=runner, persist_memory=False)
        await assistant.respond(
            "architecture source implementation",
            ground_repository=False,
        )
        self.assertNotIn(
            "[GARVIS read-only repository evidence]",
            runner.calls[0]["input"],
        )

    def test_unrelated_memories_are_not_recalled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NeuroStore(Path(directory) / "memory.db")
            store.add_memory(
                session_id="test",
                kind=MemoryKind.SEMANTIC,
                status=EvidenceStatus.SUPPLIED,
                content="Concrete finishing schedules and drywall estimates.",
                source="test",
                confidence=0.9,
                importance=0.9,
            )
            memories = store.list_memories(session_id="test")
            self.assertEqual(
                recall("Explain mitochondrial DNA inheritance.", memories),
                (),
            )

    def test_assistant_output_is_not_put_in_searchable_episode_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NeuroStore(Path(directory) / "memory.db")
            consolidate_turn(
                store,
                session_id="test",
                user_text="I want selective recall.",
                assistant_text="A very long assistant answer that must not echo.",
            )
            memories = store.list_memories(session_id="test")
            episode = next(
                memory
                for memory in memories
                if memory.kind is MemoryKind.EPISODE
            )
            self.assertIn("I want selective recall.", episode.content)
            self.assertNotIn("very long assistant answer", episode.content)


if __name__ == "__main__":
    unittest.main()
