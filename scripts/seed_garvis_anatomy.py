#!/usr/bin/env python3
"""Teach the anatomy-inspired architecture to GARVIS neurocognitive memory."""

from __future__ import annotations

import os
from pathlib import Path

from garvis.anatomical_architecture.knowledge import anatomy_software_learning_pack
from garvis.anatomical_architecture.registry import list_systems
from garvis.neurocognitive.models import EvidenceStatus, MemoryKind
from garvis.neurocognitive.store import NeuroStore


home = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
store = NeuroStore(home / "neurocognitive.db")

existing = {
    memory.content
    for memory in store.list_memories(session_id="global", limit=5_000)
}

added = 0

overview = (
    "Adrien D. Thomas supplied an anatomy-to-software architecture model for GARVIS. "
    "Human anatomy is studied at gross and microscopic levels, with organization from "
    "cells to tissues, organs, and 11 major organ systems. GARVIS should use this as a "
    "functional software analogy rather than a claim of biological equivalence."
)

if overview not in existing:
    store.add_memory(
        session_id="global",
        kind=MemoryKind.SEMANTIC,
        status=EvidenceStatus.SUPPLIED,
        content=overview,
        source="Adrien D. Thomas anatomy architecture instruction",
        confidence=0.9,
        importance=1.0,
    )
    added += 1

for definition in list_systems():
    content = (
        f"Anatomical software mapping — {definition.system.value}: "
        f"biological role: {definition.biological_role} "
        f"software role: {definition.software_role} "
        f"responsibilities: {', '.join(definition.responsibilities)}."
    )
    if content in existing:
        continue
    store.add_memory(
        session_id="global",
        kind=MemoryKind.SEMANTIC,
        status=EvidenceStatus.SUPPLIED,
        content=content,
        source="Adrien D. Thomas anatomy architecture instruction",
        confidence=0.9,
        importance=0.9,
    )
    added += 1

print(f"Added {added} anatomy-inspired GARVIS memories.")
print(f"Database: {store.path}")
print()
print(anatomy_software_learning_pack())
