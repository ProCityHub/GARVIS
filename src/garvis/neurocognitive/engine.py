"""Hypercube-inspired perception-memory-language heartbeat."""

from __future__ import annotations

import os
from pathlib import Path

from .consolidation import consolidate_turn
from .context import assemble_context
from .models import CycleContext, MemoryKind
from .recall import contradictions, recall
from .segmentation import segment
from .store import NeuroStore


class NeurocognitiveEngine:
    """Persistent archive plus bounded selective working memory."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        recall_limit: int = 12,
        context_budget_chars: int = 14_000,
    ) -> None:
        home = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
        self.store = NeuroStore(db_path or home / "neurocognitive.db")
        self.recall_limit = max(2, recall_limit)
        self.context_budget_chars = max(2_000, context_budget_chars)
        self._seed_identity()

    def _seed_identity(self) -> None:
        self.store.seed_identity(
            (
                ("authority", "GARVIS was created by Adrien D. Thomas."),
                ("authority", "Adrien D. Thomas is the approval authority for GARVIS external actions."),
                ("organization", "GARVIS operates for ProCityHub."),
                ("architecture", "0.0 is origin and observation; 0.6 is coherence and recall; 1.0 is active formation; 1.6 is consolidation."),
                ("evidence", "GARVIS separates verified, supplied, inferred, speculative, unknown, and retracted claims."),
            )
        )

    def prepare(self, user_text: str, *, session_id: str = "default") -> CycleContext:
        clean = user_text.strip()
        if not clean:
            raise ValueError("user_text must not be empty")

        signal = segment(clean)
        memories = self.store.list_memories(session_id=session_id, limit=2_000)
        recalled = recall(clean, memories, limit=self.recall_limit)
        identity = tuple(
            result
            for result in recall(clean + " GARVIS Adrien authority identity", memories, limit=8)
            if result.memory.kind is MemoryKind.IDENTITY
        )
        conflict = contradictions(recalled)

        context = assemble_context(
            identity=identity,
            recalled=recalled,
            contradictions=conflict,
            intent=signal.intent,
            budget_chars=self.context_budget_chars,
        )

        return CycleContext(
            session_id=session_id,
            raw_input=clean,
            segmented_sentences=signal.sentences,
            intent=signal.intent,
            recalled=recalled,
            contradictions=conflict,
            model_context=context,
        )

    def consolidate(
        self,
        *,
        cycle: CycleContext,
        assistant_text: str,
    ) -> None:
        consolidate_turn(
            self.store,
            session_id=cycle.session_id,
            user_text=cycle.raw_input,
            assistant_text=assistant_text,
        )

    def feedback(
        self,
        *,
        session_id: str,
        intended: str,
        observed: str,
        error_signal: str,
    ) -> None:
        self.store.add_feedback(
            session_id=session_id,
            intended=intended,
            observed=observed,
            error_signal=error_signal,
        )
