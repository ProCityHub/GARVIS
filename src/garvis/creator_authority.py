"""Creator-owned runtime authority for GARVIS.

Adrien D. Thomas is the creator and final human authority for GARVIS.

THANOS is not a GARVIS brain, identity, heartbeat, or authority source.
This module defines standing authority for automatic internal heartbeat work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Final, Union

CREATOR: Final = "Adrien D. Thomas"
PROJECT: Final = "ProCityHub/GARVIS"
AUTHORITY_SOURCE: Final = "CREATOR_DIRECTIVE"
CREATOR_ASSERTION: Final = (
    "Adrien D. Thomas is the creator and final human authority for GARVIS."
)
RUNTIME_SCOPE: Final = "garvis-runtime"


class CreatorAction(str, Enum):
    RESEARCH = "research"
    INSPECT = "inspect"
    REASON = "reason"
    PLAN = "plan"
    SIMULATE = "simulate"
    CAPTURE_PREDICTION_WITNESS = "capture-prediction-witness"
    VERIFY = "verify"
    LEARN = "learn"
    CONSOLIDATE = "consolidate"
    MONITOR = "monitor"
    TEST = "test"
    REPAIR = "repair"
    RESTART_HEARTBEAT = "restart-heartbeat"
    CONTINUE_HEARTBEAT = "continue-heartbeat"


DEFAULT_CREATOR_ACTIONS: Final = tuple(CreatorAction)


def _canonical(payload: dict) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class CreatorAuthority:
    creator: str = CREATOR
    project: str = PROJECT
    runtime_scope: str = RUNTIME_SCOPE
    authority_source: str = AUTHORITY_SOURCE
    enabled: bool = True
    allowed_actions: tuple[str, ...] = tuple(
        action.value for action in DEFAULT_CREATOR_ACTIONS
    )

    def payload(self) -> dict:
        return {
            "creator": self.creator,
            "project": self.project,
            "runtime_scope": self.runtime_scope,
            "authority_source": self.authority_source,
            "enabled": self.enabled,
            "allowed_actions": list(self.allowed_actions),
        }

    @property
    def sha256(self) -> str:
        return sha256(
            _canonical(self.payload()).encode("utf-8")
        ).hexdigest()

    def permits(self, action: Union[CreatorAction, str]) -> bool:
        value = action.value if isinstance(action, CreatorAction) else str(action)
        return (
            self.enabled
            and self.creator == CREATOR
            and self.project == PROJECT
            and self.authority_source == AUTHORITY_SOURCE
            and value in self.allowed_actions
        )


def require_creator_authority(
    authority: CreatorAuthority,
    action: Union[CreatorAction, str],
) -> None:
    if not authority.permits(action):
        value = action.value if isinstance(action, CreatorAction) else str(action)
        raise PermissionError(
            "creator authority does not permit action: " + value
        )
