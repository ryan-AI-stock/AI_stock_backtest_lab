import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.narrow_capital_latest_rs_panel_repair import run_narrow_capital_latest_rs_panel_repair


class NarrowCapitalLatestRsPanelRepairTest(unittest.TestCase):
    def test_repairs_case_rs_fields_and_blocks_non_exact_case_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "backtest_cache" / "stock_pool_observations"
            cache.mkdir(parents=True)
            dates = pd.bdate_range("2026-03-20", periods=73).strftime("%Y-%m-%d").tolist()
            self.assertEqual(dates[-1], "2026-06-30")
            for name, base, step, volume in [
                ("0050_TW.csv", 100.0, 0.3, 1000),
                ("00631L_TW.csv", 50.0, 0.6, 2000),
                ("6669_TW.csv", 200.0, 1.2, 3000),
                ("2308_TW.csv", 150.0, 0.8, 2500),
                ("2317_TW.csv", 120.0, 0.5, 2200),
            ]:
                closes = [base + idx * step for idx in range(len(dates))]
                pd.DataFrame(
                    {
                        "date": dates,
                        "close": closes,
                        "adj_close": closes,
                        "volume": [volume] * len(dates),
                    }
                ).to_csv(cache / name, index=False)

            manifest = run_narrow_capital_latest_rs_panel_repair(repo_root=root, output_dir=root / "out")

            self.assertTrue(manifest["ready_for_case_membership_rerun"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertEqual(manifest["future_data_violation_count"], 0)
            case = pd.read_csv(root / "out" / "case_ticker_rs_panel_20260630.csv")
            self.assertEqual(set(case["ticker"]), {"0050.TW", "00631L.TW", "6669.TW", "2308.TW", "2317.TW"})
            self.assertTrue(bool(case["rs_fields_ready"].all()))
            self.assertTrue(bool(case["as_of_price_available"].all()))
            row_6669 = case[case["ticker"].eq("6669.TW")].iloc[0]
            row_0050 = case[case["ticker"].eq("0050.TW")].iloc[0]
            self.assertAlmostEqual(
                float(row_6669["rs20_vs_0050_pct"]),
                float(row_6669["ret20_trailing_pct"]) - float(row_0050["ret20_trailing_pct"]),
                places=6,
            )
            self.assertEqual(row_6669["turnover_rank_scope"], "full_local_price_cache_panel_at_asof")

            trace = pd.read_csv(root / "out" / "case_trace_rs_panel_20260703.csv")
            self.assertTrue(bool(trace["case_trace_only"].all()))
            self.assertFalse(bool(trace["as_of_price_available"].any()))
            self.assertEqual(
                set(trace["blocked_reason"]),
                {"local_price_cache_not_available_for_requested_case_trace_date"},
            )
            future = pd.read_csv(root / "out" / "future_data_audit.csv")
            self.assertFalse(bool(future["future_data_violation"].any()))


if __name__ == "__main__":
    unittest.main()
