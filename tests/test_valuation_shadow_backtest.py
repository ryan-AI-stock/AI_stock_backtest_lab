from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.stock_pool_store import symbol_entry
from backtest_lab.valuation_shadow_backtest import (
    ValuationShadowVariant,
    run_valuation_shadow_backtest,
    simulate_valuation_shadow_strategy,
)


class ValuationShadowBacktestTest(unittest.TestCase):
    def test_valuation_gate_can_rotate_from_overpriced_leader_to_valid_candidate(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=190)
        prices = {
            "2317.TW": _trend_frame(dates, start=100, step=0.45),
            "2382.TW": _trend_frame(dates, start=100, step=0.20),
        }
        with tempfile.TemporaryDirectory() as tmp:
            valuation_path = Path(tmp) / "valuation.csv"
            valuation_path.write_text(
                "source_date,symbol,eps_estimate_low,eps_estimate_high,fair_pe,buy_price\n"
                "2024-07-01,2317,13,15,14,130\n"
                "2024-07-01,2382,20,22,14,500\n",
                encoding="utf-8",
            )

            baseline, baseline_diag = simulate_valuation_shadow_strategy(
                name="baseline",
                prices_by_ticker=prices,
                asset_types={"2317.TW": "stock", "2382.TW": "stock"},
                start_date="2024-08-01",
                end_date="2024-08-30",
                initial_cash=1_000_000,
                cost_model=TaiwanCostModel(),
                valuation_data=valuation_path,
                variant=ValuationShadowVariant("baseline", "原始通用排序"),
            )
            gated, gated_diag = simulate_valuation_shadow_strategy(
                name="valuation_gate",
                prices_by_ticker=prices,
                asset_types={"2317.TW": "stock", "2382.TW": "stock"},
                start_date="2024-08-01",
                end_date="2024-08-30",
                initial_cash=1_000_000,
                cost_model=TaiwanCostModel(),
                valuation_data=valuation_path,
                variant=ValuationShadowVariant("valuation_gate", "估值買點硬閘門", require_valuation_gate=True),
            )

            self.assertEqual(baseline_diag["signal_rows"][0]["target"], "2317.TW")
            self.assertEqual(gated_diag["signal_rows"][0]["target"], "2382.TW")
            self.assertGreater(baseline.final_value, 0)
            self.assertGreater(gated_diag["valuation_signal_avg_hits_per_rebalance"], 0)

    def test_runner_writes_summary_trades_and_signals(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=190)
        prices = {
            "2317.TW": _trend_frame(dates, start=100, step=0.45),
            "2382.TW": _trend_frame(dates, start=100, step=0.20),
        }
        pool = {
            "pool_id": "test_pool",
            "name": "測試股票池",
            "resolved_symbols": [
                symbol_entry("2317.TW", source="test"),
                symbol_entry("2382.TW", source="test"),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            valuation_path = Path(tmp) / "valuation.csv"
            valuation_path.write_text(
                "source_date,symbol,eps_estimate_low,eps_estimate_high,fair_pe,buy_price\n"
                "2024-07-01,2317,13,15,14,130\n",
                encoding="utf-8",
            )
            with patch(
                "backtest_lab.valuation_shadow_backtest._load_observation_price_frames",
                return_value=(prices, []),
            ):
                manifest = run_valuation_shadow_backtest(
                    pool=pool,
                    start_date="2024-08-01",
                    end_date="2024-08-30",
                    initial_cash=1_000_000,
                    cost_model=TaiwanCostModel(),
                    cache_dir=Path(tmp) / "cache",
                    output_dir=Path(tmp) / "out",
                    valuation_data=valuation_path,
                )

            root = Path(tmp) / "out"
            self.assertEqual(manifest["pool_id"], "test_pool")
            self.assertTrue((root / "valuation_shadow_summary.csv").exists())
            self.assertTrue((root / "valuation_shadow_manifest.json").exists())
            self.assertTrue((root / "valuation_shadow_report.md").exists())
            self.assertTrue((root / "valuation_gate" / "signals.csv").exists())
            self.assertTrue((root / "valuation_gate" / "trades.csv").exists())


def _trend_frame(dates: pd.DatetimeIndex, *, start: float, step: float) -> pd.DataFrame:
    closes = [start + index * step for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [20_000_000] * len(dates),
        },
        index=dates,
    )


if __name__ == "__main__":
    unittest.main()
