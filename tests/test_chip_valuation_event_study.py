from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.chip_valuation_event_study import (
    _latest_factor_row,
    build_formal_challenger_summary,
    classify_hypotheses,
    summarize_factor_events,
)


class ChipValuationEventStudyTests(unittest.TestCase):
    def test_post_profit_chip_failure_requires_profit_price_failure_and_two_factors(self) -> None:
        feature = {
            "ticker": "2454.TW",
            "holding_peak_gain_pct": 0.31,
            "drawdown_10d_pct": -0.09,
            "below_ma20": True,
            "institutional_sync_sell": True,
            "margin_overheat_flag": True,
            "short_lending_pressure_flag": False,
            "day_trading_overheat_flag": False,
            "day_trading_volume_ratio": 12.0,
            "relative_ret_10d_vs_market_pct": -0.02,
            "valuation_gate_passed": True,
            "foreign_consecutive_sell_days": 0,
            "investment_trust_net_buy_shares": 0,
        }

        classified = classify_hypotheses(feature)

        self.assertTrue(classified["post_profit_chip_failure"])
        self.assertFalse(classified["crowding_without_price_failure"])
        self.assertEqual(classified["factor_risk_count"], 2)

    def test_crowding_without_price_failure_stays_diagnostic(self) -> None:
        feature = {
            "ticker": "2330.TW",
            "holding_peak_gain_pct": 0.10,
            "drawdown_10d_pct": -0.01,
            "below_ma20": False,
            "institutional_sync_sell": False,
            "margin_overheat_flag": False,
            "short_lending_pressure_flag": True,
            "day_trading_overheat_flag": False,
            "day_trading_volume_ratio": 38.0,
            "relative_ret_10d_vs_market_pct": 0.03,
            "valuation_gate_passed": True,
            "foreign_consecutive_sell_days": 0,
            "investment_trust_net_buy_shares": 0,
        }

        classified = classify_hypotheses(feature)

        self.assertTrue(classified["crowding_without_price_failure"])
        self.assertFalse(classified["post_profit_chip_failure"])

    def test_institutional_divergence_detects_foreign_sell_trust_buy(self) -> None:
        feature = {
            "ticker": "2308.TW",
            "holding_peak_gain_pct": 0,
            "drawdown_10d_pct": 0,
            "below_ma20": False,
            "institutional_sync_sell": False,
            "margin_overheat_flag": False,
            "short_lending_pressure_flag": False,
            "day_trading_overheat_flag": False,
            "day_trading_volume_ratio": 0,
            "relative_ret_10d_vs_market_pct": 0,
            "valuation_gate_passed": True,
            "foreign_consecutive_sell_days": 4,
            "investment_trust_net_buy_shares": 1000,
        }

        classified = classify_hypotheses(feature)

        self.assertTrue(classified["institutional_divergence"])

    def test_summary_marks_valuation_without_data_as_gap(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "period_id": "bear_2022",
                    "ticker": "2454.TW",
                    "valuation_entry_block": False,
                    "valuation_data_available": False,
                    "future_ret_5d_pct": 0.0,
                    "future_ret_10d_pct": 0.0,
                    "future_ret_20d_pct": 0.0,
                    "future_rel_ret_20d_vs_market_pct": 0.0,
                    "future_max_adverse_20d_pct": 0.0,
                }
            ]
        )

        summary = summarize_factor_events(panel)
        valuation_row = summary[summary["hypothesis_id"] == "valuation_entry_block"].iloc[0]

        self.assertEqual(valuation_row["data_readiness"], "data_gap")

    def test_challenger_not_promoted_when_return_lags_baseline(self) -> None:
        baseline = pd.DataFrame(
            [
                {"period_id": "bear_2022", "total_return_pct": 10.0, "max_drawdown_pct": -5.0},
                {"period_id": "year_2023", "total_return_pct": 20.0, "max_drawdown_pct": -6.0},
                {"period_id": "ep05_2024_2026", "total_return_pct": 300.0, "max_drawdown_pct": -20.0},
            ]
        )
        challenger = pd.DataFrame(
            [
                {
                    "period_id": "bear_2022",
                    "candidate_id": "challenger_post_profit_chip_failure",
                    "total_return_pct": 9.0,
                    "max_drawdown_pct": -4.0,
                    "hypothesis_id": "post_profit_chip_failure",
                },
                {
                    "period_id": "year_2023",
                    "candidate_id": "challenger_post_profit_chip_failure",
                    "total_return_pct": 21.0,
                    "max_drawdown_pct": -5.0,
                    "hypothesis_id": "post_profit_chip_failure",
                },
                {
                    "period_id": "ep05_2024_2026",
                    "candidate_id": "challenger_post_profit_chip_failure",
                    "total_return_pct": 301.0,
                    "max_drawdown_pct": -19.0,
                    "hypothesis_id": "post_profit_chip_failure",
                },
            ]
        )
        events = pd.DataFrame(
            [
                {"hypothesis_id": "post_profit_chip_failure", "event_count": 3},
            ]
        )

        summary = build_formal_challenger_summary(
            baseline_frame=baseline,
            challenger_frame=challenger,
            event_summary=events,
        )

        status = summary.loc[
            summary["hypothesis_id"] == "post_profit_chip_failure",
            "formal_promotion_status",
        ].iloc[0]
        self.assertEqual(status, "not_promoted")

    def test_latest_factor_row_rejects_stale_factor_data(self) -> None:
        date = pd.Timestamp("2023-12-29")
        lookup = {(date, "2454.TW"): object()}

        row, factor_date = _latest_factor_row(
            lookup,
            [date],
            pd.Timestamp("2024-01-15"),
            "2454.TW",
            max_lag_days=7,
        )

        self.assertIsNone(row)
        self.assertIsNone(factor_date)


if __name__ == "__main__":
    unittest.main()
