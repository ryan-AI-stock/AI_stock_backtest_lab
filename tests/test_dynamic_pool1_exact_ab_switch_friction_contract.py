import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_exact_ab_switch_friction_contract import (
    run_dynamic_pool1_exact_ab_switch_friction_contract,
)


class DynamicPool1ExactABSwitchFrictionContractTest(unittest.TestCase):
    def test_builds_live_safe_ab_contract_with_asof_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attribution_dir = root / "attr"
            attribution_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2024-02-15",
                        "variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "incumbent_ticker": "1111.TW",
                        "challenger_ticker": "2222.TW",
                        "incumbent_holding_age_days": 6,
                        "incumbent_rank": 5,
                        "challenger_rank": 1,
                        "score_A": 0.50,
                        "score_B": 0.62,
                        "quality_score_A": 0.2,
                        "quality_score_B": 0.3,
                        "close_vs_ma20_A": 2.0,
                        "close_vs_ma20_B": 4.0,
                        "close_vs_ma60_A": 5.0,
                        "close_vs_ma60_B": 8.0,
                        "short_heat_only_flag": False,
                        "medium_quality_confirmed_flag": True,
                        "A_forward_return_5d": 1.0,
                        "B_forward_return_5d": 2.0,
                        "A_forward_return_10d": 1.0,
                        "B_forward_return_10d": 3.0,
                        "A_forward_return_20d": 1.0,
                        "B_forward_return_20d": 4.0,
                        "A_forward_return_40d": 1.0,
                        "B_forward_return_40d": 5.0,
                        "period_label": "2023_2026_requested",
                    }
                ]
            ).to_csv(attribution_dir / "ab_switch_comparison_panel.csv", index=False)
            candidate = root / "candidate.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2024-01",
                        "candidate_as_of_date": "2024-01-31",
                        "ticker": "1111",
                        "candidate_rank": 5,
                        "candidate_score": 0.50,
                        "candidate_layer": "core",
                        "price_ready_flag": True,
                        "benchmark_0050_ready_flag": True,
                        "benchmark_00631l_ready_flag": True,
                        "ret_20d_vs_0050_trailing": 1.0,
                        "ret_60d_vs_0050_trailing": 2.0,
                        "ret_20d_vs_00631L_trailing": 1.0,
                        "ret_60d_vs_00631L_trailing": 2.0,
                        "benchmark_blocked_reason": "",
                    },
                    {
                        "candidate_month": "2024-02",
                        "candidate_as_of_date": "2024-02-10",
                        "ticker": "2222",
                        "candidate_rank": 1,
                        "candidate_score": 0.62,
                        "candidate_layer": "core",
                        "price_ready_flag": True,
                        "benchmark_0050_ready_flag": True,
                        "benchmark_00631l_ready_flag": True,
                        "ret_20d_vs_0050_trailing": 3.0,
                        "ret_60d_vs_0050_trailing": 4.0,
                        "ret_20d_vs_00631L_trailing": 3.0,
                        "ret_60d_vs_00631L_trailing": 4.0,
                        "benchmark_blocked_reason": "",
                    },
                    {
                        "candidate_month": "2024-03",
                        "candidate_as_of_date": "2024-03-31",
                        "ticker": "2222",
                        "candidate_rank": 99,
                        "candidate_score": 9.99,
                        "candidate_layer": "future",
                        "price_ready_flag": True,
                        "benchmark_0050_ready_flag": True,
                        "benchmark_00631l_ready_flag": True,
                        "ret_20d_vs_0050_trailing": 99.0,
                        "ret_60d_vs_0050_trailing": 99.0,
                        "ret_20d_vs_00631L_trailing": 99.0,
                        "ret_60d_vs_00631L_trailing": 99.0,
                        "benchmark_blocked_reason": "",
                    },
                ]
            ).to_csv(candidate, index=False)
            context = root / "context.csv"
            pd.DataFrame(
                [
                    {
                        "year_month": "2024-01",
                        "available_date": "2024-01-31",
                        "ticker": "1111",
                        "name": "A",
                        "market": "TWSE",
                        "fundamental_quality_raw": 0.2,
                        "fundamentals_score": 0.2,
                        "revenue_yoy_3m_avg_pct": 2.0,
                        "liquidity_score": 0.8,
                        "avg_turnover_value": 1000,
                    },
                    {
                        "year_month": "2024-02",
                        "available_date": "2024-02-10",
                        "ticker": "2222",
                        "name": "B",
                        "market": "TWSE",
                        "fundamental_quality_raw": 0.3,
                        "fundamentals_score": 0.3,
                        "revenue_yoy_3m_avg_pct": 3.0,
                        "liquidity_score": 0.9,
                        "avg_turnover_value": 2000,
                    },
                ]
            ).to_csv(context, index=False)
            signal = root / "signal.csv"
            pd.DataFrame([{"date": "2024-02-15", "next_tradable_date": "2024-02-16"}]).to_csv(signal, index=False)

            manifest = run_dynamic_pool1_exact_ab_switch_friction_contract(
                repo_root=root,
                attribution_dir=attribution_dir,
                candidate_contract=candidate,
                candidate_context=context,
                v2_signal_panel=signal,
                output_dir=root / "out",
            )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["portfolio_replay_executed"])
            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertEqual(manifest["default_backtest_period_contract"][0]["requested_start"], "2015-01-02")
            contract = pd.read_csv(root / "out" / "exact_ab_switch_friction_contract.csv")
            self.assertEqual(contract.loc[0, "rank_B"], 1)
            self.assertEqual(contract.loc[0, "candidate_as_of_date_B"], "2024-02-10")
            self.assertTrue(bool(contract.loc[0, "combined_ab_switch_friction_strict"]))
            self.assertFalse(bool(contract.loc[0, "uses_forward_return_as_rule"]))
            readiness = pd.read_csv(root / "out" / "contract_readiness_summary.csv")
            self.assertIn("default_backtest_period_1_date_contract", set(readiness["metric"]))
            manifest_file = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest_file["ready_for_strategy_replay"])


if __name__ == "__main__":
    unittest.main()
