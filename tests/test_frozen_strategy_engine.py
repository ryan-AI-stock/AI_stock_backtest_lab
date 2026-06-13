from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.config import load_config
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.frozen_strategy_engine import load_frozen_strategy_context_from_cache, simulate_frozen_baseline
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch


class FrozenStrategyEngineTest(unittest.TestCase):
    def test_cached_context_matches_direct_frozen_baseline_simulation(self) -> None:
        source_cache = Path("backtest_cache/ad_hoc_20260612_daily_targets_filled")
        if not source_cache.exists():
            self.skipTest("local frozen strategy price cache is not available")
        config = load_config("configs/ep05_universe.json")
        group = config.group_by_id("group_c_0050_00631l_plus_mega_caps")
        labels = {asset.ticker: asset.label for asset in group.assets}
        asset_types = {asset.ticker: asset.asset_type for asset in group.assets}

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for ticker in labels:
                frame = load_price_csv(source_cache / f"{ticker.replace('.', '_')}.csv")
                frame.loc[frame.index <= pd.Timestamp("2024-03-29")].reset_index(names="date").to_csv(
                    cache_dir / f"{ticker.replace('.', '_')}.csv",
                    index=False,
                )

            context = load_frozen_strategy_context_from_cache(cache_dir=cache_dir)
            via_engine = simulate_frozen_baseline(
                context=context,
                name="via_engine",
                start_date="2024-01-02",
                end_date="2024-03-29",
                initial_cash=1_000_000,
            )

            direct_prices = {
                ticker: load_price_csv(cache_dir / f"{ticker.replace('.', '_')}.csv")
                for ticker in labels
            }
            direct_dividends = {
                ticker: split_adjusted_dividends(frame, config.manual_splits.get(ticker, ()))
                for ticker, frame in direct_prices.items()
            }
            direct = simulate_regime_mode_switch(
                name="direct",
                prices_by_ticker=direct_prices,
                asset_types=asset_types,
                market_prices=direct_prices["0050.TW"],
                start_date="2024-01-02",
                end_date="2024-03-29",
                initial_cash=1_000_000,
                cost_model=config.cost_model,
                variant=frozen_cycle_proven_top1_v1_variant(),
                dividend_series_by_ticker=direct_dividends,
            )

        self.assertAlmostEqual(via_engine.final_value, direct.final_value)
        self.assertEqual(len(via_engine.trades), len(direct.trades))
        self.assertEqual(
            via_engine.equity_curve.iloc[-1]["current_ticker"],
            direct.equity_curve.iloc[-1]["current_ticker"],
        )


if __name__ == "__main__":
    unittest.main()
