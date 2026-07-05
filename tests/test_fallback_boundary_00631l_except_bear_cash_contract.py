import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.fallback_boundary_00631l_except_bear_cash_contract import (
    run_fallback_boundary_00631l_except_bear_cash_contract,
    run_fallback_boundary_p2_bear_cash_classifier_contract,
    run_regime_conditioned_fallback_boundary_contract,
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

    def test_v2_merges_p1_p2_and_preserves_unclassified_cash_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "p1.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2021-12-29",
                        "execution_date": "2021-12-30",
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
                        "reason": "Pool1 未形成可交易正式目標。",
                        "no_target_cash_all_applied": True,
                        "source_decision": "p1",
                    }
                ]
            ).to_csv(p1, index=False)
            p2 = root / "p2.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2023-01-03",
                        "execution_date": "2023-01-04",
                        "formal_target": "CASH",
                        "formal_target_display": "cash",
                        "target_weights": "{}",
                        "no_target_reason": "pool2_disagrees_confirmation_1_not_met",
                        "risk_off_state": "no_target_cash_all",
                        "pool1_top_candidate": "00631L.TW",
                        "pool2_confirmation_state": "pool2_disagreement_confirmation_not_met",
                        "execution_action_basis": "next_day",
                        "next_day_tradable_flag": True,
                        "source_decision": "p2",
                        "readiness_state": "formal_ready",
                        "blocked_reason": "",
                    },
                    {
                        "signal_date": "2023-01-04",
                        "execution_date": "2023-01-05",
                        "formal_target": "00631L.TW",
                        "formal_target_display": "0050正二",
                        "target_weights": '{"00631L.TW": 1.0}',
                        "no_target_reason": "",
                        "risk_off_state": "formal_target_active",
                        "pool1_top_candidate": "00631L.TW",
                        "pool2_confirmation_state": "pool2_aligned_or_not_required",
                        "execution_action_basis": "next_day",
                        "next_day_tradable_flag": True,
                        "source_decision": "p2",
                        "readiness_state": "formal_ready",
                        "blocked_reason": "",
                    },
                ]
            ).to_csv(p2, index=False)
            cache = root / "backtest_cache" / "stock_pool_observations"
            cache.mkdir(parents=True)
            price_rows = [{"date": d, "close": 100.0} for d in ["2021-12-30", "2023-01-04", "2023-01-05"]]
            pd.DataFrame(price_rows).to_csv(cache / "0050_TW.csv", index=False)
            pd.DataFrame(price_rows).to_csv(cache / "00631L_TW.csv", index=False)

            manifest = run_fallback_boundary_p2_bear_cash_classifier_contract(
                repo_root=root,
                formal_streams=[p1, p2],
                output_dir=root / "out_v2",
            )

            self.assertTrue(manifest["ready_for_experiments"])
            self.assertFalse(manifest["bear_cash_classification_ready"])
            self.assertEqual(manifest["no_stock_target_but_market_exposure_allowed_rows"], 1)
            self.assertEqual(manifest["unclassified_cash_boundary_blocked_rows"], 1)

            panel = pd.read_csv(root / "out_v2" / "fallback_boundary_execution_state_panel_v2.csv")
            p2_row = panel[panel["signal_date"].eq("2023-01-03")].iloc[0]
            self.assertEqual(p2_row["execution_state"], "no_stock_target_but_market_exposure_allowed")
            blocked = panel[panel["signal_date"].eq("2021-12-29")].iloc[0]
            self.assertEqual(blocked["execution_state"], "unclassified_cash_boundary_blocked")

    def test_regime_conditioned_contract_blocks_long_strong_primary_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2023-01-02", periods=180)
            p2 = root / "p2.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": dates[140].strftime("%Y-%m-%d"),
                        "execution_date": dates[141].strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "formal_target_display": "cash",
                        "no_target_reason": "pool2_disagrees_confirmation_1_not_met",
                        "risk_off_state": "no_target_cash_all",
                        "pool1_top_candidate": "00631L.TW",
                        "pool2_confirmation_state": "pool2_disagreement_confirmation_not_met",
                        "source_decision": "p2",
                    },
                    {
                        "signal_date": dates[142].strftime("%Y-%m-%d"),
                        "execution_date": dates[143].strftime("%Y-%m-%d"),
                        "formal_target": "CASH",
                        "formal_target_display": "cash",
                        "no_target_reason": "pool1_no_actionable_formal_target",
                        "risk_off_state": "no_target_cash_all",
                        "pool1_top_candidate": "",
                        "pool2_confirmation_state": "no_pool2_persistent_eligible_candidate",
                        "source_decision": "p2",
                    },
                ]
            ).to_csv(p2, index=False)
            cache = root / "backtest_cache" / "stock_pool_observations"
            cache.mkdir(parents=True)
            price_rows = []
            for idx, date in enumerate(dates):
                price_rows.append({"date": date.strftime("%Y-%m-%d"), "close": 100 + idx})
            pd.DataFrame(price_rows).to_csv(cache / "0050_TW.csv", index=False)
            pd.DataFrame(price_rows).to_csv(cache / "00631L_TW.csv", index=False)

            manifest = run_regime_conditioned_fallback_boundary_contract(
                repo_root=root,
                formal_streams=[p2],
                output_dir=root / "out_regime",
            )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["strategy_replay_executed_by_core"])
            self.assertEqual(manifest["no_stock_target_but_market_exposure_allowed_rows"], 1)
            self.assertEqual(manifest["regime_eligible_no_stock_market_exposure_rows"], 0)
            panel = pd.read_csv(root / "out_regime" / "regime_conditioned_fallback_boundary_contract.csv")
            self.assertEqual(panel.loc[0, "market_regime_state"], "long_strong_trend")
            self.assertFalse(bool(panel.loc[0, "fallback_eligible_by_regime"]))

            mapped = pd.read_csv(root / "out_regime" / "regime_conditioned_execution_state_panel.csv")
            primary = mapped[mapped["variant_id"].eq("regime_conditioned_fallback_00631l_primary")]
            first = primary[primary["signal_date"].eq(dates[140].strftime("%Y-%m-%d"))].iloc[0]
            self.assertTrue(pd.isna(first["mapped_target"]) or first["mapped_target"] == "")
            self.assertIn("blocked_by_regime_long_strong_trend", first["action_blocked_reason"])
            blocked = primary[primary["signal_date"].eq(dates[142].strftime("%Y-%m-%d"))].iloc[0]
            self.assertTrue(pd.isna(blocked["mapped_target"]) or blocked["mapped_target"] == "")
            self.assertIn("blocked_missing_explicit_bear", blocked["action_blocked_reason"])


if __name__ == "__main__":
    unittest.main()
