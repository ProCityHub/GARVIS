"""Health-aware routing metadata for GARVIS Universal AI Router V1.1.

This module never executes a provider. Health is operational routing evidence,
not evidence that a provider's answer is true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

from .universal_ai_registry import AIOrgan, CandidateType, UniversalAIRegistry


@dataclass(frozen=True)
class ProviderHealthSnapshot:
    model: str
    failure_count: int = 0
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    blocked: bool = False


@dataclass(frozen=True)
class RoutedCandidate:
    organ_id: str
    model: Optional[str]
    provider_id: str
    blocked: bool
    failure_count: int
    has_recorded_success: bool
    reason: str


def _snapshot_for(
    model: Optional[str],
    health: Mapping[str, ProviderHealthSnapshot],
) -> ProviderHealthSnapshot:
    if not model:
        return ProviderHealthSnapshot(model="")
    return health.get(model, ProviderHealthSnapshot(model=model))


def rank_remote_candidates(
    registry: UniversalAIRegistry,
    health: Mapping[str, ProviderHealthSnapshot],
    *,
    capability: str = "text",
) -> Tuple[RoutedCandidate, ...]:
    """Rank configured programmable remotes by operational health.

    Ordering:
    1. not blocked,
    2. has a recorded success,
    3. fewer recorded failures,
    4. stable provider/model lexical tie-break.

    This is intentionally not a truth/intelligence score.
    """
    rows = []
    for organ in registry.candidates(capability):
        if organ.candidate_type is not CandidateType.REMOTE_API:
            continue
        snap = _snapshot_for(organ.model, health)
        rows.append(
            (
                (
                    1 if snap.blocked else 0,
                    0 if snap.last_success_at is not None else 1,
                    max(0, int(snap.failure_count)),
                    organ.provider_id,
                    organ.model or "",
                ),
                RoutedCandidate(
                    organ_id=organ.organ_id,
                    model=organ.model,
                    provider_id=organ.provider_id,
                    blocked=snap.blocked,
                    failure_count=max(0, int(snap.failure_count)),
                    has_recorded_success=snap.last_success_at is not None,
                    reason=(
                        "operational health ordering only; provider output remains "
                        "candidate information requiring GARVIS verification"
                    ),
                ),
            )
        )
    rows.sort(key=lambda row: row[0])
    return tuple(row[1] for row in rows)


def select_unblocked(
    ranked: Iterable[RoutedCandidate],
    *,
    limit: int = 3,
) -> Tuple[RoutedCandidate, ...]:
    selected = []
    for item in ranked:
        if item.blocked:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return tuple(selected)
