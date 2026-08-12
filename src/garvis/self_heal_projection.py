from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .self_heal_root import sha256_file


@dataclass(frozen=True)
class TrustedEntry:
    path: str
    sha256: str
    baseline: str
    auto_repair: bool = False


@dataclass(frozen=True)
class DriftObservation:
    path: str
    expected_sha256: str
    actual_sha256: str
    baseline: str
    drifted: bool


@dataclass(frozen=True)
class RepairDecision:
    observation: DriftObservation
    disposition: str
    auto_repair: bool


MISSING_SHA256 = "MISSING"
TRUSTED = "TRUSTED"
REPAIR_REQUIRED = "REPAIR_REQUIRED"
REPORT_ONLY = "REPORT_ONLY"



def build_plan(root: Path, entries: Mapping[str, TrustedEntry]) -> tuple[RepairDecision, ...]:
    decisions: list[RepairDecision] = []

    for relative_path, entry in sorted(entries.items()):
        target = root / relative_path
        actual_sha256 = sha256_file(target) if target.is_file() else MISSING_SHA256
        drifted = actual_sha256 != entry.sha256
        if not drifted:
            disposition = TRUSTED
        elif entry.auto_repair:
            disposition = REPAIR_REQUIRED
        else:
            disposition = REPORT_ONLY
        decisions.append(
            RepairDecision(
                observation=DriftObservation(
                    path=entry.path,
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_sha256,
                    baseline=entry.baseline,
                    drifted=drifted,
                ),
                disposition=disposition,
                auto_repair=entry.auto_repair,
            )
        )

    return tuple(decisions)
