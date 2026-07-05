import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_incumbent_protection_bounded_diagnostic_runner import (
    run_dynamic_pool1_incumbent_protection_bounded_diagnostic,
)


class DynamicPool1IncumbentProtectionBoundedDiagnosticTest(unittest.TestCase):
    def test_suppresses_switch_when_incumbent_rule_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = root / "signal.csv"
            dates = pd.date_range("2024-01-02", periods=8, freq="B")
            pd.DataFrame(
                [
                    {
                        "date": dates[0].strftime("%Y-%m-%d"),
                        "next_tradable_date": dates[1].strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "formal_state": "no_target_cash",
                        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "dynamic_selected_canonical_ticker": "1111.TW",
                        "dynamic_blocked_reason": "",
                    },
                    {
                        "date": dates[1].strftime("%Y-%m-%d"),
                        "next_tradable_date": dates[2].strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "formal_state": "no_target_cash",
                        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "dynamic_selected_canonical_ticker": "2222.TW",
                        "dynamic_blocked_reason": "",
                    },
                    {
                        "date": dates[2].strftime("%Y-%m-%d"),
                        "next_tradable_date": dates[3].strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "formal_state": "no_target_cash",
                        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "dynamic_selected_canonical_ticker": "",
                        "dynamic_blocked_reason": "blocked_no_asof_dynamic_candidate_pool",
                    },
                ]
            ).to_csv(signal, index=False)
            rules = root / "rules.csv"
            pd.DataFrame(
                [
                    {
                        "switch_event_id": "s1",
                        "date": dates[1].strftime("%Y-%m-%d"),
                        "variant_id": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "rule_id": "v2_keep_A_if_still_working_top5_unless_B_score10",
                        "rule_candidate_triggered": True,
                        "incumbent_ticker_A": "1111.TW",
                        "challenger_ticker_B": "2222.TW",
                        "incumbent_A_still_working_flag": True,
                        "incumbent_A_trend_break_flag": False,
                        "A_rank_still_top5": True,
                        "score_margin": 0.05,
                        "rank_margin": 2,
                        "B_minus_A_forward_delta_20d": -1.0,
                        "B_minus_A_forward_delta_40d": -2.0,
                        "forward_return_used_as_evaluation_metadata": True,
                        "uses_forward_return_as_rule": False,
                        "future_data_violation": False,
                    },
                    {
                        "switch_event_id": "s1",
                        "date": dates[1].strftime("%Y-%m-%d"),
                        "variant_id": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "rule_id": "v2_keep_A_if_still_working_top10_and_no_trend_break",
                        "rule_candidate_triggered": False,
                        "incumbent_ticker_A": "1111.TW",
                        "challenger_ticker_B": "2222.TW",
                        "uses_forward_return_as_rule": False,
                        "future_data_violation": False,
                    },
                ]
            ).to_csv(rules, index=False)
            formal_dir = root / "outputs" / "combined_formal_target_stream_20150128_20211230_20260702"
            formal_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "signal_date": d.strftime("%Y-%m-%d"),
                        "execution_date": d.strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    }
                    for d in dates
                ]
            ).to_csv(formal_dir / "combined_formal_target_stream.csv", index=False)
            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            price_rows = []
            for i, d in enumerate(dates):
                for ticker, close in [("1111", 100 + i), ("2222", 80 + i)]:
                    price_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": ticker, "market": "TWSE", "close": close})
            pd.DataFrame(price_rows).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)
            bench = root / "backtest_cache" / "stock_pool_observations"
            bench.mkdir(parents=True)
            for name in ["0050_TW.csv", "00631L_TW.csv"]:
                pd.DataFrame([{"date": d.strftime("%Y-%m-%d"), "close": 100 + i} for i, d in enumerate(dates)]).to_csv(
                    bench / name, index=False
                )

            manifest = run_dynamic_pool1_incumbent_protection_bounded_diagnostic(
                repo_root=root,
                signal_panel=signal,
                rule_contract=rules,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertEqual(manifest["future_data_violation_count"], 0)
            suppression = pd.read_csv(root / "out" / "switch_suppression_ledger.csv")
            self.assertIn("incumbent_protection_top5_unless_B_score10_primary", set(suppression["variant"]))
            primary = suppression[suppression["variant"].eq("incumbent_protection_top5_unless_B_score10_primary")].iloc[0]
            self.assertEqual(primary["incumbent_ticker_A"], "1111.TW")
            self.assertEqual(primary["challenger_ticker_B"], "2222.TW")
            self.assertEqual(primary["current_dynamic_selected_ticker"], "2222.TW")
            execution = pd.read_csv(root / "out" / "execution_state_daily_panel.csv")
            self.assertIn("market_exposure_fallback_ticker", execution.columns)


if __name__ == "__main__":
    unittest.main()
