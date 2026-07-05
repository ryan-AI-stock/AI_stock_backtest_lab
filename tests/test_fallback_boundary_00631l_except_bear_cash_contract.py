import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.fallback_boundary_00631l_except_bear_cash_contract import (
    run_fallback_boundary_00631l_except_bear_cash_contract,
)


class FallbackBoundary00631LExceptBearCashContractTest(unittest.TestCase):
    def test_classifies_only_traceable_market_exposure_cash_for_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = root / "formal.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "execution_date": "2024-01-03",
                        "formal_target": "CASH",
                        "formal_target_display": "cash",
                        "target_type": "risk_control_cash",
                        "pool1_candidate": "00631L.TW",
                        "pool1_candidate_display": "0050正二",
                        "pool1_gate_status": "has_formal_pool1_target",
                        "pool1_attack_gate_active": False,
                        "pool1_target_is_actionable": True,
                        "pool2_confirmation_status": "pool2_not_ready",
                        "pool2_confirmation_state": "no_pool2_persistent_eligible_candidate",
                        "no_target_reason": "pool2_confirmation_not_ready",
                        "risk_off_state": "no_target_cash_all",
                        "reason": "Pool1 有主攻目標，但 Pool2 持續確認未通過，啟動風險控管空手。",
                        "no_target_cash_all_applied": True,
                        "source_decision": "test",
                    },
                    {
                        "signal_date": "2024-01-03",
                        "execution_date": "2024-01-04",
                        "formal_target": "CASH",
                        "formal_target_display": "cash",
                        "target_type": "risk_control_cash",
                        "pool1_candidate": "",
                        "pool1_candidate_display": "",
                        "pool1_gate_status": "no_actionable_pool1_target",
                        "pool1_attack_gate_active": False,
                        "pool1_target_is_actionable": False,
                        "pool2_confirmation_status": "pool2_not_ready",
                        "pool2_confirmation_state": "no_pool2_persistent_eligible_candidate",
                        "no_target_reason": "pool1_no_actionable_formal_target",
                        "risk_off_state": "no_target_cash_all",
                        "reason": "Pool1 未形成可交易正式目標，啟動風險控管空手。",
                        "no_target_cash_all_applied": True,
                        "source_decision": "test",
                    },
                    {
                        "signal_date": "2024-01-04",
                        "execution_date": "2024-01-05",
                        "formal_target": "2330.TW",
                        "formal_target_display": "台積電",
                        "target_type": "stock",
                        "pool1_candidate": "2330.TW",
                        "pool1_candidate_display": "台積電",
                        "pool1_gate_status": "has_formal_pool1_target",
                        "pool1_attack_gate_active": True,
                        "pool1_target_is_actionable": True,
                        "pool2_confirmation_status": "confirmed_by_pool2_persistence",
                        "pool2_confirmation_state": "pool2_persistence_ready",
                        "no_target_reason": "",
                        "risk_off_state": "formal_target_active",
                        "reason": "",
                        "no_target_cash_all_applied": False,
                        "source_decision": "test",
                    },
                ]
            ).to_csv(stream, index=False)
            cache = root / "backtest_cache" / "stock_pool_observations"
            cache.mkdir(parents=True)
            price_rows = [{"date": d, "close": 100.0} for d in ["2024-01-03", "2024-01-04", "2024-01-05"]]
            pd.DataFrame(price_rows).to_csv(cache / "0050_TW.csv", index=False)
            pd.DataFrame(price_rows).to_csv(cache / "00631L_TW.csv", index=False)

            manifest = run_fallback_boundary_00631l_except_bear_cash_contract(
                repo_root=root,
                formal_stream=stream,
                output_dir=root / "out",
            )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["bear_cash_classification_ready"])
            self.assertEqual(manifest["no_stock_target_but_market_exposure_allowed_rows"], 1)
            self.assertEqual(manifest["unclassified_cash_boundary_blocked_rows"], 1)

            daily = pd.read_csv(root / "out" / "fallback_mapping_daily_panel.csv")
            primary = daily[daily["variant_id"].eq("fallback_00631L_except_bear_cash_primary")]
            self.assertEqual(primary.loc[primary["signal_date"].eq("2024-01-02"), "mapped_target"].iloc[0], "00631L.TW")
            blocked = primary[primary["signal_date"].eq("2024-01-03")].iloc[0]
            self.assertTrue(pd.isna(blocked["mapped_target"]) or blocked["mapped_target"] == "")
            self.assertIn("blocked_missing_explicit_bear", blocked["action_blocked_reason"])

            upper = daily[daily["variant_id"].eq("fallback_00631L_all_no_target_upper_bound_reference")]
            self.assertTrue((upper[upper["formal_target"].eq("CASH")]["mapped_target"] == "00631L.TW").all())


if __name__ == "__main__":
    unittest.main()
