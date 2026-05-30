import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.simulation import (
    simulate_buy_and_hold,
    simulate_dual_momentum_vol_control,
    simulate_relative_strength_top1,
)


def price_frame(open_price: float, close_start: float, daily_step: float, rows: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    closes = [close_start + (daily_step * index) for index in range(rows)]
    frame = pd.DataFrame(
        {
            "open": [open_price for _ in dates],
            "close": closes,
            "adj_close": closes,
        },
        index=dates,
    )
    frame.loc[pd.Timestamp("2024-01-02"), "open"] = open_price
    frame.loc[pd.Timestamp("2024-01-02"), "close"] = close_start + daily_step * (len(frame.loc[: "2024-01-02"]) - 1)
    frame.loc[pd.Timestamp("2024-01-03"), "open"] = open_price
    return frame.sort_index()


class SimulationTest(unittest.TestCase):
    def test_benchmark_buys_own_asset_on_first_trade_date(self) -> None:
        prices = price_frame(100, 100, 1)
        result = simulate_buy_and_hold(
            name="0050 買進持有",
            ticker="0050.TW",
            asset_type="etf",
            prices=prices,
            start_date="2024-01-02",
            end_date="2024-01-03",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )

        self.assertEqual(result.trades[0].date, "2024-01-02")
        self.assertEqual(result.trades[0].ticker, "0050.TW")
        self.assertEqual(result.trades[0].action, "buy")
        self.assertEqual(result.trades[0].reason, "benchmark_initial_entry")

    def test_strategy_initial_entry_uses_previous_signal_not_fixed_benchmark(self) -> None:
        flat_0050 = price_frame(100, 100, 0)
        strong_stock = price_frame(200, 100, 5)
        prices_by_ticker = {
            "0050.TW": flat_0050,
            "6669.TW": strong_stock,
        }

        result = simulate_relative_strength_top1(
            name="相對強弱第一名",
            prices_by_ticker=prices_by_ticker,
            asset_types={"0050.TW": "etf", "6669.TW": "stock"},
            start_date="2024-01-02",
            end_date="2024-01-03",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )

        self.assertEqual(result.trades[0].date, "2024-01-02")
        self.assertEqual(result.trades[0].ticker, "6669.TW")
        self.assertEqual(result.trades[0].reason, "relative_strength_initial_entry")

    def test_dual_momentum_uses_professional_style_filters_and_rebalances_weekly(self) -> None:
        flat_0050 = price_frame(100, 100, 0)
        strong_stock = price_frame(200, 100, 3)
        prices_by_ticker = {
            "0050.TW": flat_0050,
            "6669.TW": strong_stock,
        }

        result = simulate_dual_momentum_vol_control(
            name="雙動能波動控管",
            prices_by_ticker=prices_by_ticker,
            asset_types={"0050.TW": "etf", "6669.TW": "stock"},
            start_date="2024-01-02",
            end_date="2024-01-31",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )

        self.assertEqual(result.trades[0].date, "2024-01-02")
        self.assertEqual(result.trades[0].ticker, "6669.TW")
        self.assertEqual(result.trades[0].reason, "dual_momentum_initial_entry")
        self.assertLessEqual(len([trade for trade in result.trades if trade.action in {"buy", "sell"}]), 2)
        self.assertIn("current_ticker", result.equity_curve.columns)

    def test_dual_momentum_can_rebalance_monthly(self) -> None:
        flat_0050 = price_frame(100, 100, 0)
        strong_stock = price_frame(200, 100, 3)
        prices_by_ticker = {
            "0050.TW": flat_0050,
            "6669.TW": strong_stock,
        }

        result = simulate_dual_momentum_vol_control(
            name="雙動能月頻",
            prices_by_ticker=prices_by_ticker,
            asset_types={"0050.TW": "etf", "6669.TW": "stock"},
            start_date="2024-01-02",
            end_date="2024-02-29",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            rebalance_frequency="monthly",
        )

        self.assertEqual(result.trades[0].ticker, "6669.TW")
        self.assertIn("current_ticker", result.equity_curve.columns)


if __name__ == "__main__":
    unittest.main()
