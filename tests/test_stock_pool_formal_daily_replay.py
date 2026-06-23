from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_formal_daily_replay import run_stock_pool_formal_daily_replay


class StockPoolFormalDailyReplayTest(unittest.TestCase):
    def test_replay_trades_only_on_consensus_and_keeps_position_on_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "out"
            cache.mkdir()
            replay_panel = root / "panel.csv"
            pd.DataFrame(
                [
                    _pool_row("2024-01-02", "ai_theme_large_cap_v20260613", "00631L.TW", True),
                    _pool_row("2024-01-02", "tw50_dynamic_constituents_v0", "00631L.TW", True),
                    _pool_row("2024-01-02", "large_core_bluechip_v0", "2882.TW", True),
                    _pool_row("2024-01-03", "ai_theme_large_cap_v20260613", "2330.TW", True),
                    _pool_row("2024-01-03", "tw50_dynamic_constituents_v0", "2882.TW", True),
                    _pool_row("2024-01-03", "large_core_bluechip_v0", "00631L.TW", True),
                    _pool_row("2024-01-04", "ai_theme_large_cap_v20260613", "2882.TW", True),
                    _pool_row("2024-01-04", "tw50_dynamic_constituents_v0", "2882.TW", True),
                    _pool_row("2024-01-04", "large_core_bluechip_v0", "00631L.TW", True),
                ]
            ).to_csv(replay_panel, index=False)
            _write_price(cache / "00631L_TW.csv", [("2024-01-02", 10.0), ("2024-01-03", 11.0), ("2024-01-04", 12.0)])
            _write_price(cache / "2882_TW.csv", [("2024-01-02", 50.0), ("2024-01-03", 51.0), ("2024-01-04", 52.0)])
            _write_price(cache / "2330_TW.csv", [("2024-01-02", 600.0), ("2024-01-03", 610.0), ("2024-01-04", 620.0)])

            result = run_stock_pool_formal_daily_replay(
                replay_panel_path=replay_panel,
                price_cache_dir=cache,
                output_dir=output,
                initial_cash=100_000,
            )

            daily = pd.read_csv(result / "baseline_three_pool_formal_daily_equity.csv")
            self.assertEqual(daily.loc[0, "winner_ticker"], "00631L.TW")
            self.assertEqual(daily.loc[0, "position_ticker"], "00631L.TW")
            self.assertGreater(float(daily.loc[0, "transaction_cost"]), 0)
            self.assertEqual(daily.loc[1, "consensus_state"], "divergent")
            self.assertEqual(daily.loc[1, "position_ticker"], "00631L.TW")
            self.assertEqual(daily.loc[1, "action"], "hold")
            self.assertEqual(daily.loc[2, "winner_ticker"], "2882.TW")
            self.assertEqual(daily.loc[2, "position_ticker"], "2882.TW")
            self.assertEqual(daily.loc[2, "action"], "switch")
            self.assertGreater(float(daily.loc[2, "transaction_cost"]), 0)
            self.assertTrue((result / "formal_three_pool_summary.csv").exists())


def _pool_row(date: str, pool_id: str, ticker: str, eligible: bool) -> dict:
    return {
        "period": "2024",
        "signal_date": date,
        "pool_id": pool_id,
        "top_ticker": ticker,
        "eligible_for_pool_selection": eligible,
    }


def _write_price(path: Path, rows: list[tuple[str, float]]) -> None:
    pd.DataFrame(
        [
            {
                "date": date,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adj_close": price,
                "volume": 1000,
            }
            for date, price in rows
        ]
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
