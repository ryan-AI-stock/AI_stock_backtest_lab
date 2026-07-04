import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.strong_stock_trend_extension_daily_event_contract import (
    run_strong_stock_trend_extension_daily_event_contract,
)


class StrongStockTrendExtensionDailyEventContractTest(unittest.TestCase):
    def test_builds_exact_daily_trend_extension_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "outputs" / "dynamic_pool1_benchmark_aware_candidate_contract_20260704"
            context_dir = root / "outputs" / "dynamic_pool1_candidate_panel_v0_20260704"
            candidate_dir.mkdir(parents=True)
            context_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2024-01",
                        "candidate_as_of_date": "2024-01-02",
                        "ticker": "2330",
                        "candidate_rank": 1,
                        "candidate_score": 1.0,
                        "candidate_layer": "core",
                        "benchmark_filter_primary_selected": True,
                    }
                ]
            ).to_csv(candidate_dir / "dynamic_pool1_benchmark_aware_candidate_contract.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "year_month": "2024-01",
                        "ticker": "2330",
                        "name": "台積電",
                        "market": "TWSE",
                        "candidate_layer": "core",
                        "selected_for_pool_v0": True,
                    }
                ]
            ).to_csv(context_dir / "candidate_pool_by_month.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            dates = pd.date_range("2023-07-03", periods=160, freq="B")
            rows = []
            for idx, date in enumerate(dates):
                close = 100 + idx
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": "2330",
                        "name": "台積電",
                        "market": "TWSE",
                        "close": close,
                        "turnover_value": 100_000_000 + idx * 2_000_000,
                        "liquidity_pass": True,
                    }
                )
            pd.DataFrame(rows).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)

            bench_dir = root / "backtest_cache" / "stock_pool_observations"
            bench_dir.mkdir(parents=True)
            for name in ["0050_TW.csv", "00631L_TW.csv"]:
                pd.DataFrame(
                    [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "open": 100,
                            "high": 100,
                            "low": 100,
                            "close": 100 + idx * 0.1,
                            "adj_close": 100 + idx * 0.1,
                            "volume": 1,
                            "dividend": 0,
                            "stock_split": 0,
                        }
                        for idx, date in enumerate(dates)
                    ]
                ).to_csv(bench_dir / name, index=False)

            manifest = run_strong_stock_trend_extension_daily_event_contract(
                repo_root=root,
                candidate_contract=candidate_dir / "dynamic_pool1_benchmark_aware_candidate_contract.csv",
                candidate_context=context_dir / "candidate_pool_by_month.csv",
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )
            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertFalse(manifest["uses_cross_section_median_as_primary_benchmark"])
            events = pd.read_csv(root / "out" / "trend_extension_daily_event_contract.csv")
            self.assertFalse(events.empty)
            self.assertIn("trend_ext_ma_stack_breakout", set(events["event_variant"]))
            self.assertFalse(events["uses_forward_return_as_rule"].any())


if __name__ == "__main__":
    unittest.main()
