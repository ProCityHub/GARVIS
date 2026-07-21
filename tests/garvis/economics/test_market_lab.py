from garvis.economics.market_lab import PositionPlan, bond_price, position_size


def test_position_size_uses_defined_risk() -> None:
    assert position_size(PositionPlan(10_000.0, 0.01, 100.0, 95.0)) == 20.0


def test_bond_price_equals_face_when_coupon_equals_yield() -> None:
    assert round(bond_price(1_000.0, 0.05, 0.05, 10), 6) == 1_000.0
