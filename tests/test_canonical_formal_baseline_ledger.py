import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.canonical_formal_baseline_ledger import run_canonical_formal_baseline_ledger


class CanonicalFormalBaselineLedgerTest(unittest.TestCase):
    def test_builds_next_day_ledger_with_cash_mapping_and_costs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = root / "formal.csv"
            pd.DataFrame(
                [
                    {"signal_date": "2024-01-02", "execution_date": "2024-01-03", "formal_target": "CASH"},
                    {"signal_date": "2024-01-03", "execution_date": "2024-01-04", "formal_target": "2330.TW"},
                    {"signal_date": "2024-01-04", "execution_date": "2024-01-05", "formal_target": "2330.TW"},
                    {"signal_date": "2024-01-05", "execution_date": "2024-01-08", "formal_target": "CASH"},
                ]
            ).to_csv(stream, index=False)
            cache = root / "backtest_cache" / "stock_pool_observations"
            cache.mkdir(parents=True)
            dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
            for ticker, prices in {
                "2330_TW.csv": [100, 100, 110, 120],
                "0050_TW.csv": [50, 51, 52, 53],
                "00631L_TW.csv": [20, 21, 22, 23],
            }.items():
                pd.DataFrame({"date": dates, "close": prices, "adj_close": prices}).to_csv(cache / ticker, index=False)

            manifest = run_canonical_formal_baseline_ledger(
                repo_root=root,
                formal_streams=[stream],
                output_dir=root / "out",
            )

            self.assertEqual(manifest["execution_basis"], "next_day")
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertTrue(manifest["ready_for_experiments"])
            daily = pd.read_csv(root / "out" / "canonical_formal_daily_ledger.csv")
            self.assertEqual(daily.loc[0, "held_ticker"], "CASH")
            self.assertEqual(daily.loc[1, "held_ticker"], "2330.TW")
            self.assertEqual(daily.loc[1, "canonical_current_formal_mapping"], "target_100pct")
            trades = pd.read_csv(root / "out" / "canonical_cost_trade_summary.csv")
            self.assertGreater(int(trades.loc[0, "trade_rows"]), 0)
            reset = pd.read_csv(root / "out" / "canonical_period_reset_summary.csv")
            self.assertIn("2024_latest", set(reset["period_label"]))
            future = pd.read_csv(root / "out" / "future_data_audit.csv")
            self.assertFalse(bool(future["future_data_violation"].any()))


if __name__ == "__main__":
    unittest.main()
