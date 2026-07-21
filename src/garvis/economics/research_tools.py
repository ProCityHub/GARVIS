"""Agents SDK tools for bounded economic research."""

from __future__ import annotations

import json
import os
from typing import Any

from agents import function_tool

from .bitcoin_lab import MiningInputs, estimate_mining
from .internet_field import read_url
from .knowledge_pack import learning_context
from .market_lab import PositionPlan, bond_price, position_size


@function_tool
def economic_learning_pack(topic: str) -> str:
    """Read GARVIS's curated ProCityHub revenue, job, market, bond, and Bitcoin learning pack."""

    return learning_context(topic)


@function_tool
def research_read_url(url: str) -> str:
    """Read one public allowlisted webpage using a side-effect-free HTTP GET."""

    return json.dumps(read_url(url), ensure_ascii=False)


@function_tool
def calculate_bitcoin_mining_case(
    hashrate_hs: float,
    network_difficulty: float,
    reward_btc: float,
    btc_price: float,
    power_watts: float,
    electricity_price_per_kwh: float,
    uptime: float = 1.0,
    pool_fee: float = 0.0,
) -> str:
    """Calculate a hypothetical Bitcoin mining case; this does not mine or access a wallet."""

    result = estimate_mining(
        MiningInputs(
            hashrate_hs=hashrate_hs,
            network_difficulty=network_difficulty,
            reward_btc=reward_btc,
            btc_price=btc_price,
            power_watts=power_watts,
            electricity_price_per_kwh=electricity_price_per_kwh,
            uptime=uptime,
            pool_fee=pool_fee,
        )
    )
    return json.dumps(result.__dict__, sort_keys=True)


@function_tool
def calculate_paper_market_case(
    account_value: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    bond_face_value: float,
    bond_coupon_rate: float,
    bond_yield_rate: float,
    bond_years: int,
) -> str:
    """Calculate paper position size and a plain bond price; no trade is placed."""

    units = position_size(
        PositionPlan(
            account_value=account_value,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=stop_price,
        )
    )
    price = bond_price(
        face_value=bond_face_value,
        annual_coupon_rate=bond_coupon_rate,
        yield_rate=bond_yield_rate,
        years=bond_years,
    )
    return json.dumps({"paper_position_units": units, "educational_bond_price": price})


def build_research_tools() -> list[Any]:
    """Attach tools only when the operator explicitly enables research mode."""

    enabled = os.getenv("GARVIS_ENABLE_RESEARCH", "").strip().casefold() in {"1", "true", "yes", "on"}
    if not enabled:
        return []
    return [
        economic_learning_pack,
        research_read_url,
        calculate_bitcoin_mining_case,
        calculate_paper_market_case,
    ]
