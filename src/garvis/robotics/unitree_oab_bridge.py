"""Governed GARVIS <-> Unitree OAB bridge.

Unitree state -> ROS2/CycloneDDS -> Observer -> GARVIS
GARVIS candidate -> OAB governance -> ROS2/CycloneDDS -> Unitree

Physical actuation is disabled by default.
CAPABILITY IS NOT AUTHORIZATION.
SIMULATION IS NOT EVIDENCE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol, Sequence


class BridgeMode(str, Enum):
    OBSERVE = "observe"
    SIMULATION = "simulation"
    ACTUATE = "actuate"


@dataclass(frozen=True)
class UnitreeCommand:
    action: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class UnitreeObservation:
    state: Mapping[str, Any]
    signature: str


@dataclass(frozen=True)
class BridgeResult:
    executed: bool
    simulated: bool
    reason: str
    observation_signature: str
    transport_result: Mapping[str, Any] | None = None


class UnitreeTransport(Protocol):
    def read_state(self) -> Mapping[str, Any]: ...
    def publish(self, command: UnitreeCommand) -> Mapping[str, Any]: ...


AuthorizationCheck = Callable[[UnitreeCommand, str | None], bool]


def _signature(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


class UnitreeOABBridge:
    def __init__(
        self,
        transport: UnitreeTransport,
        *,
        mode: BridgeMode = BridgeMode.OBSERVE,
        external_action_allowed: bool = False,
        authorization_check: AuthorizationCheck | None = None,
    ) -> None:
        self.transport = transport
        self.mode = mode
        self.external_action_allowed = external_action_allowed
        self.authorization_check = authorization_check

    def observe(self) -> UnitreeObservation:
        state = dict(self.transport.read_state())
        return UnitreeObservation(state=state, signature=_signature(state))

    def execute(
        self,
        command: UnitreeCommand,
        *,
        legal_actions: Sequence[str] | None = None,
        approval_token: str | None = None,
    ) -> BridgeResult:
        before = self.observe()
        if legal_actions is not None and command.action not in set(legal_actions):
            return BridgeResult(False, False, "LEGAL_ACTION_MASK_REJECT", before.signature)
        if self.mode is BridgeMode.OBSERVE:
            return BridgeResult(False, False, "OBSERVE_ONLY", before.signature)
        if self.mode is BridgeMode.SIMULATION:
            return BridgeResult(False, True, "SIMULATION_ONLY", before.signature)
        if not self.external_action_allowed:
            return BridgeResult(False, False, "EXTERNAL_ACTION_DISABLED", before.signature)
        if self.authorization_check is None:
            return BridgeResult(False, False, "AUTHORIZATION_CHECK_MISSING", before.signature)
        if not self.authorization_check(command, approval_token):
            return BridgeResult(False, False, "AUTHORIZATION_REJECTED", before.signature)
        result = dict(self.transport.publish(command))
        return BridgeResult(True, False, "AUTHORIZED_TRANSPORT_PUBLISH", before.signature, result)
