from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.sector_dynamic_pool_backtest import (
    SectorPoolVariant,
    load_theme_members,
    score_candidates,
    simulate_sector_pool,
    target_weights_from_scores,
)


class SectorDynamicPoolBacktestTest(unittest.TestCase):
    def test_load_theme_members_uses_primary_theme_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theme_map.csv"
            pd.DataFrame(
                [
                    {"theme": "記憶體", "symbol": "2408", "name": "南亞科", "role": "DRAM", "conviction": "high", "primary": "yes"},
                    {"theme": "記憶體", "symbol": "9999", "name": "排除", "role": "", "conviction": "low", "primary": "no"},
                ]
            ).to_csv(path, index=False)
            members = load_theme_members(path, "記憶體")
        self.assertEqual([member.ticker for member in members], ["2408.TW"])

    def test_target_weights_keep_single_stock_cap_and_cash(self) -> None:
        weights = target_weights_from_scores(
            [("A.TW", 1.0), ("B.TW", 0.8), ("C.TW", 0.7)],
            SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.30),
        )
        self.assertEqual(weights, {"A.TW": 0.30, "B.TW": 0.30, "C.TW": 0.30})
        self.assertLess(sum(weights.values()), 1.0)

    def test_score_candidates_uses_signal_date_not_future_prices(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=160)
        base = pd.Series(range(100, 260), index=dates, dtype=float)
        future_jump = base.copy()
        future_jump.iloc[-1] = 1000.0
        prices = {
            "A.TW": _price_frame(base),
            "B.TW": _price_frame(future_jump),
        }
        before_jump_scores = score_candidates(prices, dates[-2], SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.3, min_avg_turnover_twd=1))
        after_jump_scores = score_candidates(prices, dates[-1], SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.3, min_avg_turnover_twd=1))
        self.assertEqual(before_jump_scores[0][0], "B.TW")
        self.assertNotEqual(before_jump_scores[0][1], after_jump_scores[0][1])

    def test_simulate_sector_pool_produces_multiholding_equity(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=220)
        prices = {
            "A.TW": _price_frame(pd.Series(range(100, 320), index=dates, dtype=float)),
            "B.TW": _price_frame(pd.Series(range(90, 310), index=dates, dtype=float)),
            "C.TW": _price_frame(pd.Series(range(80, 300), index=dates, dtype=float)),
        }
        result = simulate_sector_pool(
            variant=SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.30, min_avg_turnover_twd=1),
            prices_by_ticker=prices,
            labels={ticker: ticker for ticker in prices},
            asset_types={ticker: "stock" for ticker in prices},
            start_date="2023-08-01",
            end_date="2023-10-31",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )
        self.assertGreater(result.result.final_value, 0)
        self.assertIn("market_exposure", result.result.equity_curve.columns)
        self.assertFalse(result.holdings.empty)


def _price_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
            "dividend": 0.0,
        },
        index=close.index,
    )


if __name__ == "__main__":
    unittest.main()
