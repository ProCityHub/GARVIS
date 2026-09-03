from garvis.robotics.unitree_oab_bridge import BridgeMode, UnitreeCommand, UnitreeOABBridge


class FakeTransport:
    def __init__(self):
        self.published = []

    def read_state(self):
        return {"mode": "stand", "position": [0.0, 0.0, 0.0]}

    def publish(self, command):
        self.published.append(command)
        return {"accepted": True}


def test_observe_and_default_no_actuation():
    transport = FakeTransport()
    bridge = UnitreeOABBridge(transport)
    assert len(bridge.observe().signature) == 64
    result = bridge.execute(UnitreeCommand("stand", {}), legal_actions=["stand"])
    assert not result.executed
    assert result.reason == "OBSERVE_ONLY"
    assert transport.published == []


def test_simulation_never_publishes():
    transport = FakeTransport()
    bridge = UnitreeOABBridge(transport, mode=BridgeMode.SIMULATION)
    result = bridge.execute(UnitreeCommand("move", {"x": 0.1}), legal_actions=["move"])
    assert result.simulated and not result.executed
    assert transport.published == []


def test_legal_mask_is_hard_gate():
    transport = FakeTransport()
    bridge = UnitreeOABBridge(
        transport,
        mode=BridgeMode.ACTUATE,
        external_action_allowed=True,
        authorization_check=lambda command, token: True,
    )
    result = bridge.execute(UnitreeCommand("move", {}), legal_actions=["stand"], approval_token="ok")
    assert not result.executed
    assert result.reason == "LEGAL_ACTION_MASK_REJECT"
    assert transport.published == []


def test_live_publish_requires_explicit_authorization():
    transport = FakeTransport()
    bridge = UnitreeOABBridge(
        transport,
        mode=BridgeMode.ACTUATE,
        external_action_allowed=True,
        authorization_check=lambda command, token: token == "AUTHORIZED",
    )
    rejected = bridge.execute(UnitreeCommand("stand", {}), legal_actions=["stand"], approval_token="wrong")
    assert not rejected.executed
    accepted = bridge.execute(UnitreeCommand("stand", {}), legal_actions=["stand"], approval_token="AUTHORIZED")
    assert accepted.executed
    assert len(transport.published) == 1
