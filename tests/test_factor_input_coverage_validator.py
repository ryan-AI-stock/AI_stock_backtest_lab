from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.factor_input_coverage_validator import (
    FactorSourceSpec,
    latest_not_after,
    validate_factor_source,
)


class FactorInputCoverageValidatorTests(unittest.TestCase):
    def test_latest_not_after_uses_prior_or_same_date(self) -> None:
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")]

        self.assertEqual(latest_not_after(dates, pd.Timestamp("2024-01-05")), pd.Timestamp("2024-01-05"))
        self.assertEqual(latest_not_after(dates, pd.Timestamp("2024-01-04")), pd.Timestamp("2024-01-02"))
        self.assertIsNone(latest_not_after(dates, pd.Timestamp("2024-01-01")))

    def test_daily_source_marks_stale_rows_as_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "institutional.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2023-12-29",
                        "ticker": "2454.TW",
                        "foreign_net_buy_shares": -1,
                        "investment_trust_net_buy_shares": 0,
                        "dealer_net_buy_shares": 0,
                        "foreign_consecutive_sell_days": 1,
                        "trust_consecutive_sell_days": 0,
                    }
                ]
            ).to_csv(source, index=False)
            spec = FactorSourceSpec(
                factor_id="institutional_flows",
                source_path=str(source),
                required_columns=(
                    "date",
                    "ticker",
                    "foreign_net_buy_shares",
                    "investment_trust_net_buy_shares",
                    "dealer_net_buy_shares",
                    "foreign_consecutive_sell_days",
                    "trust_consecutive_sell_days",
                ),
                max_lag_days=7,
            )

            summary, gaps = validate_factor_source(
                spec=spec,
                expected_tickers=["2454.TW"],
                trading_dates=[pd.Timestamp("2024-01-10")],
                start_date="2024-01-02",
                end_date="2024-01-10",
            )

        self.assertEqual(summary["readiness_status"], "blocked")
        self.assertEqual(summary["fresh_symbol_date_count"], 0)
        self.assertIn("source_ends_before_required_end", summary["blocked_reason"])
        self.assertEqual(gaps[0]["missing_trading_dates"], 1)

    def test_point_in_time_valuation_after_period_is_not_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "valuation.csv"
            pd.DataFrame(
                [
                    {
                        "source_date": "2026-06-14",
                        "ticker": "2454.TW",
                        "eps_estimate_low": 75,
                        "eps_estimate_high": 80,
                    }
                ]
            ).to_csv(source, index=False)
            spec = FactorSourceSpec(
                factor_id="valuation",
                source_path=str(source),
                required_columns=("source_date", "ticker", "eps_estimate_low", "eps_estimate_high"),
                max_lag_days=180,
                source_kind="manual_or_point_in_time_snapshot",
            )

            summary, _ = validate_factor_source(
                spec=spec,
                expected_tickers=["2454.TW"],
                trading_dates=[pd.Timestamp("2026-05-26")],
                start_date="2024-01-02",
                end_date="2026-05-26",
            )

        self.assertEqual(summary["readiness_status"], "blocked")
        self.assertEqual(summary["fresh_coverage_ratio"], 0.0)
        self.assertIn("source_starts_after_required_end", summary["blocked_reason"])

    def test_complete_daily_source_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "day.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "ticker": "2454.TW",
                        "day_trading_volume_ratio": 10,
                        "day_trading_ratio_5d_avg": 8,
                        "day_trading_ratio_20d_avg": 7,
                        "day_trading_overheat_flag": False,
                    },
                    {
                        "date": "2024-01-03",
                        "ticker": "2454.TW",
                        "day_trading_volume_ratio": 11,
                        "day_trading_ratio_5d_avg": 9,
                        "day_trading_ratio_20d_avg": 8,
                        "day_trading_overheat_flag": False,
                    },
                ]
            ).to_csv(source, index=False)
            spec = FactorSourceSpec(
                factor_id="day_trading",
                source_path=str(source),
                required_columns=(
                    "date",
                    "ticker",
                    "day_trading_volume_ratio",
                    "day_trading_ratio_5d_avg",
                    "day_trading_ratio_20d_avg",
                    "day_trading_overheat_flag",
                ),
                max_lag_days=7,
            )

            summary, gaps = validate_factor_source(
                spec=spec,
                expected_tickers=["2454.TW"],
                trading_dates=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                start_date="2024-01-02",
                end_date="2024-01-03",
            )

        self.assertEqual(summary["readiness_status"], "ready")
        self.assertEqual(summary["fresh_coverage_ratio"], 1.0)
        self.assertEqual(gaps, [])


if __name__ == "__main__":
    unittest.main()
