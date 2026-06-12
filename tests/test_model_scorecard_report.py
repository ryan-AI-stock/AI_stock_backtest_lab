from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.model_scorecard_report import (
    REPORT_NAME,
    ScorecardReport,
    ScorecardRow,
    build_scorecard_report,
    _normalized_equity_values,
    public_cutoff_date,
    report_filename,
    resolve_scorecard_data_end,
    write_scorecard_outputs,
)


class ModelScorecardReportTest(unittest.TestCase):
    def test_public_cutoff_uses_seven_day_delay(self) -> None:
        self.assertEqual(public_cutoff_date("2026-06-12", 7), "2026-06-05")

    def test_resolve_scorecard_data_end_uses_latest_common_date_before_cutoff(self) -> None:
        prices = {
            "0050.TW": _prices(["2026-05-29", "2026-06-04", "2026-06-05"], 100),
            "2454.TW": _prices(["2026-05-29", "2026-06-04"], 200),
        }

        self.assertEqual(resolve_scorecard_data_end(prices, "2026-05-29", "2026-06-05"), "2026-06-04")

    def test_build_scorecard_report_contains_model_and_three_comparisons(self) -> None:
        tickers = ["0050.TW", "00631L.TW", "2330.TW", "2454.TW", "2308.TW", "2317.TW", "2382.TW", "3231.TW", "6669.TW"]
        prices = {ticker: _prices(pd.date_range("2025-01-02", periods=380, freq="B"), 100 + index * 10) for index, ticker in enumerate(tickers)}
        labels = {
            "0050.TW": "0050",
            "00631L.TW": "0050正二",
            "2330.TW": "台積電",
            "2454.TW": "聯發科",
            "2308.TW": "台達電",
            "2317.TW": "鴻海",
            "2382.TW": "廣達",
            "3231.TW": "緯創",
            "6669.TW": "緯穎",
        }
        asset_types = {ticker: "stock" for ticker in tickers}
        asset_types["0050.TW"] = "etf"
        asset_types["00631L.TW"] = "etf"

        report = build_scorecard_report(
            prices_by_ticker=prices,
            labels=labels,
            asset_types=asset_types,
            manual_splits={},
            cost_model=TaiwanCostModel(),
            report_date="2026-06-12",
            tracking_start="2026-05-29",
            data_end="2026-06-10",
            initial_cash=1_000_000,
            delay_days=7,
            tracking_case_ticker="auto",
        )

        self.assertEqual(report.report_name, REPORT_NAME)
        self.assertEqual(len(report.rows), 3)
        self.assertTrue(report.model_tracking_label.startswith("AI模型追蹤："))
        self.assertIn(report.model_tracking_label, {row.name for row in report.rows})
        self.assertEqual(report.tracking_case_ticker, report.model_holding_records[-1]["ticker"])
        self.assertEqual(report.public_cutoff_date, "2026-06-05")

    def test_build_scorecard_ignores_rows_after_data_end(self) -> None:
        tickers = ["0050.TW", "00631L.TW", "2330.TW", "2454.TW", "2308.TW", "2317.TW", "2382.TW", "3231.TW", "6669.TW"]
        dates = pd.date_range("2025-01-02", periods=380, freq="B")
        prices = {ticker: _prices(dates, 100 + index * 10) for index, ticker in enumerate(tickers)}
        labels = {
            "0050.TW": "0050",
            "00631L.TW": "0050正二",
            "2330.TW": "台積電",
            "2454.TW": "聯發科",
            "2308.TW": "台達電",
            "2317.TW": "鴻海",
            "2382.TW": "廣達",
            "3231.TW": "緯創",
            "6669.TW": "緯穎",
        }
        asset_types = {ticker: "stock" for ticker in tickers}
        asset_types["0050.TW"] = "etf"
        asset_types["00631L.TW"] = "etf"
        base_report = build_scorecard_report(
            prices_by_ticker=prices,
            labels=labels,
            asset_types=asset_types,
            manual_splits={},
            cost_model=TaiwanCostModel(),
            report_date="2026-06-12",
            tracking_start="2026-05-29",
            data_end="2026-06-05",
            initial_cash=1_328_709,
            delay_days=7,
            tracking_case_ticker="auto",
        )
        future_prices = {ticker: frame.copy() for ticker, frame in prices.items()}
        future_date = pd.Timestamp("2026-06-12")
        for ticker, frame in future_prices.items():
            frame.loc[future_date] = {
                "open": 9999.0,
                "close": 9999.0,
                "adj_close": 9999.0,
                "dividend": 0.0,
                "stock_split": 0.0,
            }
            future_prices[ticker] = frame.sort_index()

        future_report = build_scorecard_report(
            prices_by_ticker=future_prices,
            labels=labels,
            asset_types=asset_types,
            manual_splits={},
            cost_model=TaiwanCostModel(),
            report_date="2026-06-12",
            tracking_start="2026-05-29",
            data_end="2026-06-05",
            initial_cash=1_328_709,
            delay_days=7,
            tracking_case_ticker="auto",
        )

        self.assertEqual(base_report.to_dict()["rows"], future_report.to_dict()["rows"])
        self.assertEqual(base_report.model_holding_records, future_report.model_holding_records)

    def test_write_scorecard_outputs_creates_latest_pdf_and_csv(self) -> None:
        row = {
            "date": "2026-05-29",
            "total_value_twd": 1_000_000,
        }
        from backtest_lab.model_scorecard_report import ScorecardReport, ScorecardRow

        report = ScorecardReport(
            report_name=REPORT_NAME,
            report_version="vtest",
            report_date="2026-06-12",
            public_cutoff_date="2026-06-05",
            data_end_date="2026-06-05",
            tracking_start_date="2026-05-29",
            initial_cash_twd=1_000_000,
            delay_days=7,
            model_name="AI大型權值股最佳版 v20260605",
            tracking_case_ticker="2454.TW",
            tracking_case_label="聯發科",
            model_tracking_label="AI模型追蹤：聯發科",
            model_holding_records=[{"start_date": "2026-05-29", "ticker": "2454.TW", "label": "聯發科", "exposure_pct": 100.0}],
            rows=[
                ScorecardRow("AI模型追蹤：聯發科", "model", 1_010_000, 1.0, -1.0, 1),
                ScorecardRow("0050買進持有", "0050.TW", 1_000_000, 0.0, 0.0, 1),
            ],
            equity_curves={"AI模型追蹤：聯發科": [row], "0050買進持有": [row]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_scorecard_outputs(output_dir, report)

            self.assertTrue((output_dir / "model_scorecard_summary.csv").exists())
            self.assertTrue((output_dir / report_filename("pdf", latest=True)).exists())
            self.assertTrue((output_dir / report_filename("md", "2026-06-12")).exists())

    def test_chart_values_start_from_same_initial_cash(self) -> None:
        rows = [
            {"date": "2026-05-29", "total_value_twd": 962_903.0},
            {"date": "2026-06-05", "total_value_twd": 980_000.0},
        ]
        values = _normalized_equity_values(rows, 1_000_000)

        self.assertEqual(values[0], 100.0)
        self.assertAlmostEqual(values[1], 101.7756, places=3)



def _prices(dates, base: float) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates)
    values = [base + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "open": values,
            "close": [value + 0.5 for value in values],
            "adj_close": [value + 0.5 for value in values],
            "dividend": [0.0] * len(index),
            "stock_split": [0.0] * len(index),
        },
        index=index,
    )


if __name__ == "__main__":
    unittest.main()
