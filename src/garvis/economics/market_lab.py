"""Paper-only stock and bond mathematics for GARVIS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionPlan:
    account_value: float
    risk_fraction: float
    entry_price: float
    stop_price: float

    def __post_init__(self) -> None:
        if self.account_value <= 0.0:
            raise ValueError("account_value must be greater than zero")
        if not 0.0 < self.risk_fraction <= 1.0:
            raise ValueError("risk_fraction must be between zero and one")
        if self.entry_price <= 0.0 or self.stop_price < 0.0:
            raise ValueError("prices must be non-negative and entry must be greater than zero")
        if self.entry_price == self.stop_price:
            raise ValueError("entry_price and stop_price must differ")


def position_size(plan: PositionPlan) -> float:
    """Return maximum units for a paper position based on defined risk."""

    risk_budget = plan.account_value * plan.risk_fraction
    risk_per_unit = abs(plan.entry_price - plan.stop_price)
    return risk_budget / risk_per_unit


def compound_value(principal: float, annual_rate: float, years: float, periods_per_year: int = 1) -> float:
    if principal < 0.0:
        raise ValueError("principal must not be negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least one")
    if years < 0.0:
        raise ValueError("years must not be negative")
    return principal * (1.0 + annual_rate / periods_per_year) ** (periods_per_year * years)


def bond_price(face_value: float, annual_coupon_rate: float, yield_rate: float, years: int, payments_per_year: int = 2) -> float:
    """Price a plain fixed-rate bond; educational use only."""

    if face_value <= 0.0:
        raise ValueError("face_value must be greater than zero")
    if years < 1 or payments_per_year < 1:
        raise ValueError("years and payments_per_year must be positive")
    periods = years * payments_per_year
    coupon = face_value * annual_coupon_rate / payments_per_year
    period_yield = yield_rate / payments_per_year
    if period_yield == 0.0:
        return face_value + coupon * periods
    coupons = sum(coupon / (1.0 + period_yield) ** period for period in range(1, periods + 1))
    principal = face_value / (1.0 + period_yield) ** periods
    return coupons + principal
