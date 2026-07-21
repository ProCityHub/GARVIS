"""Bitcoin proof-of-work and profitability mathematics for education and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECONDS_PER_DAY = 86_400
DIFFICULTY_HASHES = 2**32


@dataclass(frozen=True)
class MiningInputs:
    hashrate_hs: float
    network_difficulty: float
    reward_btc: float
    btc_price: float
    power_watts: float
    electricity_price_per_kwh: float
    uptime: float = 1.0
    pool_fee: float = 0.0
    daily_hardware_cost: float = 0.0
    daily_cooling_cost: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = (
            "hashrate_hs",
            "network_difficulty",
            "reward_btc",
            "btc_price",
            "power_watts",
            "electricity_price_per_kwh",
            "daily_hardware_cost",
            "daily_cooling_cost",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if self.network_difficulty == 0.0:
            raise ValueError("network_difficulty must be greater than zero")
        if not 0.0 <= self.uptime <= 1.0:
            raise ValueError("uptime must be between 0.0 and 1.0")
        if not 0.0 <= self.pool_fee < 1.0:
            raise ValueError("pool_fee must be between 0.0 and 1.0")


@dataclass(frozen=True)
class MiningEstimate:
    expected_blocks_per_day: float
    expected_btc_per_day: float
    gross_revenue_per_day: float
    electricity_cost_per_day: float
    total_cost_per_day: float
    net_profit_per_day: float
    break_even_btc_price: Optional[float]


def estimate_mining(inputs: MiningInputs) -> MiningEstimate:
    expected_blocks = (
        inputs.hashrate_hs * SECONDS_PER_DAY / (inputs.network_difficulty * DIFFICULTY_HASHES)
    )
    expected_btc = expected_blocks * inputs.reward_btc * inputs.uptime * (1.0 - inputs.pool_fee)
    gross_revenue = expected_btc * inputs.btc_price
    electricity_cost = (inputs.power_watts / 1_000.0) * 24.0 * inputs.electricity_price_per_kwh
    total_cost = electricity_cost + inputs.daily_hardware_cost + inputs.daily_cooling_cost
    net_profit = gross_revenue - total_cost
    break_even = None if expected_btc <= 0.0 else total_cost / expected_btc
    return MiningEstimate(
        expected_blocks_per_day=expected_blocks,
        expected_btc_per_day=expected_btc,
        gross_revenue_per_day=gross_revenue,
        electricity_cost_per_day=electricity_cost,
        total_cost_per_day=total_cost,
        net_profit_per_day=net_profit,
        break_even_btc_price=break_even,
    )
