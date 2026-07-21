from garvis.economics.bitcoin_lab import MiningInputs, estimate_mining


def test_mining_estimate_accounts_for_power_cost() -> None:
    result = estimate_mining(
        MiningInputs(
            hashrate_hs=1_000_000_000_000,
            network_difficulty=100_000_000,
            reward_btc=3.125,
            btc_price=100_000.0,
            power_watts=3_000.0,
            electricity_price_per_kwh=0.1,
        )
    )
    assert result.electricity_cost_per_day == 7.2
    assert result.total_cost_per_day == 7.2
