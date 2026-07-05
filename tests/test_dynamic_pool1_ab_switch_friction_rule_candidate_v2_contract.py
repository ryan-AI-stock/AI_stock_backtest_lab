import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract import (
    run_dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract,
)


class DynamicPool1ABSwitchFrictionRuleCandidateV2ContractTest(unittest.TestCase):
    def test_builds_v2_rule_families_and_period_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact = root / "exact.csv"
            pd.DataFrame(
                [
                    {
                        "switch_event_id": "s1",
                        "date": "2024-02-15",
                        "next_tradable_date": "2024-02-16",
                        "period_label": "2023_2026_requested",
                        "variant_id": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "top1_or_top3_source": "top1",
                        "incumbent_ticker_A": "1111.TW",
                        "challenger_ticker_B": "2222.TW",
                        "incumbent_holding_age_days": 6,
                        "rank_A": 4,
                        "rank_B": 1,
                        "rank_margin": 3,
                        "score_A": 0.5,
                        "score_B": 0.62,
                        "score_margin": 0.12,
                        "rs60_B_minus_A_vs_0050": 1.0,
                        "rs60_B_minus_A_vs_00631l": 1.0,
                        "quality_margin": 0.1,
                        "close_vs_ma20_A": 2.0,
                        "close_vs_ma60_A": 3.0,
                        "deviation_gap_B_minus_A_ma20": 4.0,
                        "deviation_gap_B_minus_A_ma60": 6.0,
                        "B_more_overheated_ma20": False,
                        "B_more_overheated_ma60": False,
                        "switch_margin_rank2_score5": True,
                        "switch_margin_rank3_score10": True,
                        "switch_no_short_heat_only": True,
                        "switch_after_min_hold5": True,
                        "short_heat_only": False,
                        "medium_quality_confirmed": True,
                        "rs_superiority": True,
                        "quality_not_lower": True,
                        "combined_ab_switch_friction_strict": True,
                        "B_minus_A_forward_delta_20d": 1.0,
                        "B_minus_A_forward_delta_40d": 2.0,
                        "future_data_violation": False,
                    },
                    {
                        "switch_event_id": "s2",
                        "date": "2024-03-01",
                        "next_tradable_date": "2024-03-04",
                        "period_label": "2023_2026_requested",
                        "variant_id": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "top1_or_top3_source": "top1",
                        "incumbent_ticker_A": "3333.TW",
                        "challenger_ticker_B": "4444.TW",
                        "incumbent_holding_age_days": 2,
                        "rank_A": 4,
                        "rank_B": 2,
                        "rank_margin": 2,
                        "score_A": 0.5,
                        "score_B": 0.56,
                        "score_margin": 0.06,
                        "rs60_B_minus_A_vs_0050": 0.0,
                        "rs60_B_minus_A_vs_00631l": 0.0,
                        "quality_margin": 0.0,
                        "close_vs_ma20_A": -1.0,
                        "close_vs_ma60_A": 2.0,
                        "deviation_gap_B_minus_A_ma20": 7.0,
                        "deviation_gap_B_minus_A_ma60": 6.0,
                        "B_more_overheated_ma20": True,
                        "B_more_overheated_ma60": False,
                        "switch_margin_rank2_score5": True,
                        "switch_margin_rank3_score10": False,
                        "switch_no_short_heat_only": True,
                        "switch_after_min_hold5": False,
                        "short_heat_only": False,
                        "medium_quality_confirmed": False,
                        "rs_superiority": True,
                        "quality_not_lower": True,
                        "combined_ab_switch_friction_strict": False,
                        "B_minus_A_forward_delta_20d": -1.0,
                        "B_minus_A_forward_delta_40d": -2.0,
                        "future_data_violation": False,
                    },
                ]
            ).to_csv(exact, index=False)
            manifest = run_dynamic_pool1_ab_switch_friction_rule_candidate_v2_contract(
                repo_root=root,
                exact_contract=exact,
                output_dir=root / "out",
            )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["portfolio_replay_executed"])
            self.assertEqual(manifest["default_backtest_period_contract"][1]["requested_end"], "2026-06-30")
            contract = pd.read_csv(root / "out" / "exact_ab_switch_friction_rule_candidate_v2_contract.csv")
            self.assertEqual(contract["rule_id"].nunique(), 10)
            balanced = contract[contract["rule_id"].eq("v2_balanced_A_working_or_B_large_margin")]
            self.assertTrue(bool(balanced.loc[balanced["switch_event_id"].eq("s1"), "rule_candidate_triggered"].iloc[0]))
            self.assertFalse(bool(contract["uses_forward_return_as_rule"].any()))
            self.assertIn("incumbent_A_still_working_flag", contract.columns)
            self.assertIn("switch_allowed_only_if_A_breaks_or_B_large_margin", contract.columns)
            self.assertIn("top_k_rank_stability_5d", contract.columns)
            manifest_file = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest_file["future_execution_state_note"]["not_applied_in_this_task"])
            readiness = pd.read_csv(root / "out" / "v2_readiness_by_rule.csv")
            self.assertIn("v2_balanced_A_working_or_B_large_margin", set(readiness["rule_id"]))
            self.assertTrue((root / "out" / "incumbent_working_context.csv").exists())
            self.assertTrue((root / "out" / "top_k_strength_context.csv").exists())


if __name__ == "__main__":
    unittest.main()
