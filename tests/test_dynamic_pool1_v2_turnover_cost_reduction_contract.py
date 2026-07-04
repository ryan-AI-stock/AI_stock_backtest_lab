import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from backtest_lab.dynamic_pool1_v2_turnover_cost_reduction_contract import (
    run_dynamic_pool1_v2_turnover_cost_reduction_contract,
)


class DynamicPool1V2TurnoverCostReductionContractTest(unittest.TestCase):
    def test_monthly_lock_outputs_required_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2024-02-01",
                        "next_tradable_date": "2024-02-02",
                        "formal_target": "CASH",
                        "formal_state": "no_target_cash",
                        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "dynamic_candidate_pool_month": "2024-01",
                    },
                    {
                        "date": "2024-02-02",
                        "next_tradable_date": "2024-02-05",
                        "formal_target": "2454.TW",
                        "formal_state": "direct_stock_target",
                        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
                        "dynamic_candidate_pool_month": "2024-01",
                    },
                ]
            ).to_csv(source / "daily_signal_panel.csv", index=False)

            member = root / "member.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2024-01",
                        "candidate_as_of_date": "2024-01-31",
                        "ticker": "2330",
                        "candidate_rank": 1,
                        "candidate_score": 1.0,
                        "candidate_layer": "core",
                        "price_ready_flag": True,
                        "benchmark_0050_ready_flag": True,
                        "benchmark_00631l_ready_flag": True,
                        "ret_60d_vs_0050_trailing": 1.0,
                        "ret_60d_vs_00631L_trailing": 1.0,
                        "ret_20d_vs_0050_trailing": 1.0,
                        "ret_20d_vs_00631L_trailing": 1.0,
                        "rs60_positive_vs_both": True,
                        "rs20_and_rs60_positive_vs_both": True,
                        "top10_and_rs60_positive_vs_both": True,
                        "benchmark_blocked_reason": "",
                        "uses_cross_section_median_as_primary_benchmark": False,
                        "forward_return_used_as_contract_rule": False,
                        "variant_id": "v2_primary_rs60_top15_monthly",
                    }
                ]
            ).to_csv(member, index=False)
            v0 = root / "v0.csv"
            pd.DataFrame([{"year_month": "2024-01", "ticker": "2330", "market": "TWSE"}]).to_csv(v0, index=False)
            liquidity = root / "liquidity" / "shards"
            liquidity.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-02-02", "ticker": "2330", "market": "TWSE", "close": 100.0},
                    {"date": "2024-02-05", "ticker": "2330", "market": "TWSE", "close": 101.0},
                ]
            ).to_csv(liquidity / "accepted_liquidity_rows_2024_02.csv", index=False)
            with mock.patch(
                "backtest_lab.dynamic_pool1_v2_turnover_cost_reduction_contract.VARIANTS",
                [
                    {
                        "variant": "v2_top15_top1_monthly_lock_when_formal_cash_or_market_exposure",
                        "source_variant_id": "v2_primary_rs60_top15_monthly",
                        "top_n": 1,
                        "rule": "monthly_lock",
                        "rank_improvement_required": 0,
                        "min_hold_days": 0,
                        "cooldown_days": 0,
                        "monthly_lock_active": True,
                        "description": "test",
                    }
                ],
            ):
                manifest = run_dynamic_pool1_v2_turnover_cost_reduction_contract(
                    repo_root=root,
                    source_contract_dir=source,
                    v2_member_panel=member,
                    candidate_v0_pool=v0,
                    liquidity_dir=root / "liquidity",
                    output_dir=root / "out",
                )
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertEqual(manifest["future_data_violation_count"], 0)
            weights = pd.read_csv(root / "out" / "daily_weight_ledger.csv")
            self.assertIn("dynamic_switch_reason", weights.columns)
            self.assertIn("v2_top15_top1_monthly_lock_when_formal_cash_or_market_exposure", set(weights["variant"]))
            overlap = pd.read_csv(root / "out" / "formal_target_overlap_audit.csv")
            self.assertFalse(overlap["formal_direct_stock_target_override_allowed"].any())


if __name__ == "__main__":
    unittest.main()
