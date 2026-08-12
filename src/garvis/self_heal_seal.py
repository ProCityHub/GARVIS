from __future__ import annotations

from dataclasses import replace

from .self_heal_projection import RepairDecision

SEALED_REPAIR_REQUIRED = "SEALED_REPAIR_REQUIRED"



def force_sealed_decision(decision: RepairDecision) -> RepairDecision:
    return replace(decision, disposition=SEALED_REPAIR_REQUIRED, auto_repair=True)
