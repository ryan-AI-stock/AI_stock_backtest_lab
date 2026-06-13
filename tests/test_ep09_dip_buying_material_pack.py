from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.ep09_dip_buying_material_pack import STRATEGIES, simulate_ladder_strategy


class Ep09DipBuyingMaterialPackTest(unittest.TestCase):
    def test_risk_filtered_ladder_blocks_adds_in_falling_long_trend(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=260)
        close = [100 - index * 0.25 for index in range(len(dates))]
        prices = pd.DataFrame(
            {
                "open": close,
                "high": [value * 1.01 for value in close],
                "low": [value * 0.99 for value in close],
                "close": close,
                "adj_close": close,
                "volume": [100_000] * len(dates),
            },
            index=dates,
        )
        dividends = pd.Series(0.0, index=dates)
        blind = next(strategy for strategy in STRATEGIES if strategy.strategy_id == "blind_dip_ladder")
        filtered = next(strategy for strategy in STRATEGIES if strategy.strategy_id == "ai_risk_filtered_ladder")

        blind_result = simulate_ladder_strategy(
            ticker="0050.TW",
            asset_label="0050",
            asset_type="etf",
            prices=prices,
            dividend_series=dividends,
            start_date="2022-01-03",
            end_date="2022-12-30",
            initial_cash=1_000_000,
            strategy=blind,
            broker_fee_rate=0.001425,
            broker_fee_discount=1.0,
            minimum_fee_twd=20,
            sell_tax_rate=0.001,
        )
        filtered_result = simulate_ladder_strategy(
            ticker="0050.TW",
            asset_label="0050",
            asset_type="etf",
            prices=prices,
            dividend_series=dividends,
            start_date="2022-01-03",
            end_date="2022-12-30",
            initial_cash=1_000_000,
            strategy=filtered,
            broker_fee_rate=0.001425,
            broker_fee_discount=1.0,
            minimum_fee_twd=20,
            sell_tax_rate=0.001,
        )

        self.assertGreater(blind_result["buy_count"], filtered_result["buy_count"])
        self.assertGreater(filtered_result["blocked_buy_days"], 0)
        self.assertGreater(filtered_result["final_value"], blind_result["final_value"])


if __name__ == "__main__":
    unittest.main()
