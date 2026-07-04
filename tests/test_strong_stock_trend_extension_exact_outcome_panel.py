import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.strong_stock_trend_extension_exact_outcome_panel import (
    run_strong_stock_trend_extension_exact_outcome_panel,
)


class StrongStockTrendExtensionExactOutcomePanelTest(unittest.TestCase):
    def test_builds_outcome_panel_with_benchmark_excess_and_incomplete_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "events"
            event_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "next_tradable_date": "2024-01-03",
                        "ticker": "2330",
                        "candidate_name": "台積電",
                        "candidate_source": "unit_test",
                        "candidate_layer": "core",
                        "event_variant": "trend_ext_ma_stack_breakout",
                        "uses_forward_return_as_rule": False,
                    },
                    {
                        "signal_date": "2024-02-01",
                        "next_tradable_date": "2024-02-02",
                        "ticker": "2330",
                        "candidate_name": "台積電",
                        "candidate_source": "unit_test",
                        "candidate_layer": "core",
                        "event_variant": "trend_ext_slope_acceleration",
                        "uses_forward_return_as_rule": False,
                    },
                ]
            ).to_csv(event_dir / "trend_extension_daily_event_contract.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            dates = pd.date_range("2024-01-02", periods=70, freq="B")
            pd.DataFrame(
                [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": "2330",
                        "close": 100 + idx,
                    }
                    for idx, date in enumerate(dates)
                ]
            ).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)

            bench_dir = root / "backtest_cache" / "stock_pool_observations"
            bench_dir.mkdir(parents=True)
            for name in ["0050_TW.csv", "00631L_TW.csv"]:
                pd.DataFrame(
                    [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "close": 100 + idx * 0.5,
                        }
                        for idx, date in enumerate(dates)
                    ]
                ).to_csv(bench_dir / name, index=False)

            manifest = run_strong_stock_trend_extension_exact_outcome_panel(
                repo_root=root,
                event_contract=event_dir / "trend_extension_daily_event_contract.csv",
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )
            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            panel = pd.read_csv(root / "out" / "trend_extension_exact_event_outcome_panel.csv")
            self.assertIn("event_return_60d_pct", panel.columns)
            self.assertIn("excess_vs_0050_20d_pct", panel.columns)
            self.assertTrue(panel["horizon_5d_complete"].any())
            self.assertTrue((~panel["all_horizons_complete"]).any())
            incomplete = pd.read_csv(root / "out" / "incomplete_outcome_rows.csv")
            self.assertFalse(incomplete.empty)


if __name__ == "__main__":
    unittest.main()
