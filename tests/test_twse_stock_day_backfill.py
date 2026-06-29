import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.twse_stock_day_backfill import normalize_twse_stock_day_payload, run_twse_stock_day_backfill


class TwseStockDayBackfillTest(unittest.TestCase):
    def test_normalizes_twse_stock_day_payload(self) -> None:
        frame = normalize_twse_stock_day_payload(_payload("103/11/03"), ticker="00631L", source_month="2014-11")

        self.assertEqual(frame["date"].iloc[0], "2014-11-03")
        self.assertEqual(frame["ticker"].iloc[0], "00631L.TW")
        self.assertEqual(frame["open"].iloc[0], 10.5)
        self.assertEqual(frame["volume"].iloc[0], 1234567.0)
        self.assertEqual(frame["source_type"].iloc[0], "official_real_price")
        self.assertEqual(frame["adjustment_policy"].iloc[0], "twse_raw_close_as_adj_close_pending_distribution_review")

    def test_runs_monthly_backfill_with_completed_and_failed_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"

            def fake_fetcher(ticker: str, month: pd.Period) -> dict:
                if str(month) == "2014-12":
                    return {"stat": "很抱歉，沒有符合條件的資料!"}
                return _payload(f"{month.year - 1911:03d}/{month.month:02d}/03")

            run_twse_stock_day_backfill(
                ticker="00631L",
                start_month="2014-11",
                end_month="2014-12",
                output_dir=output,
                fetcher=fake_fetcher,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial_completed_with_failed_months")
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["synthetic_used"])
            self.assertFalse(manifest["strategy_ready"])

            normalized = pd.read_csv(output / "00631l_201411_201412_twse_stock_day_normalized.csv")
            self.assertEqual(len(normalized), 1)
            self.assertEqual(normalized["source"].iloc[0], "TWSE_STOCK_DAY")

            completed = pd.read_csv(output / "completed.csv")
            failed = pd.read_csv(output / "failed.csv")
            self.assertEqual(set(completed["month"]), {"2014-11"})
            self.assertEqual(set(failed["month"]), {"2014-12"})

            coverage = pd.read_csv(output / "00631l_price_coverage_after_backfill.csv")
            self.assertFalse(bool(coverage["formal_ready_for_price_only"].iloc[0]))
            self.assertEqual(coverage["missing_months"].iloc[0], "2014-12")


def _payload(date: str) -> dict:
    return {
        "stat": "OK",
        "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"],
        "data": [[date, "1,234,567", "12,345,670", "10.50", "10.80", "10.30", "10.70"]],
    }


if __name__ == "__main__":
    unittest.main()
