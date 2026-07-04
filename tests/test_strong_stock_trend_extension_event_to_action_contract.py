import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.strong_stock_trend_extension_event_to_action_contract import (
    run_strong_stock_trend_extension_event_to_action_contract,
)


class StrongStockTrendExtensionEventToActionContractTest(unittest.TestCase):
    def test_builds_bounded_actions_without_formal_override_or_reference_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome_dir = root / "outcome"
            outcome_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "next_tradable_date": "2024-01-03",
                        "ticker": "2330",
                        "candidate_name": "台積電",
                        "candidate_source": "dynamic_pool1",
                        "candidate_layer": "core",
                        "event_variant": "trend_ext_slope_acceleration",
                        "case_trace_only": False,
                        "uses_forward_return_as_rule": False,
                        "entry_price": 100.0,
                        "event_return_20d_pct": 10.0,
                        "event_return_40d_pct": 20.0,
                        "excess_vs_0050_60d_pct": 3.0,
                        "excess_vs_00631L_60d_pct": -1.0,
                        "rs20_vs_0050": 5.0,
                        "rs20_vs_00631L": 4.0,
                    },
                    {
                        "signal_date": "2024-01-03",
                        "next_tradable_date": "2024-01-04",
                        "ticker": "2308",
                        "candidate_name": "台達電",
                        "candidate_source": "dynamic_pool1",
                        "candidate_layer": "core",
                        "event_variant": "trend_ext_new_high_rs_confirm",
                        "case_trace_only": False,
                        "uses_forward_return_as_rule": False,
                        "entry_price": 50.0,
                        "event_return_20d_pct": 5.0,
                        "event_return_40d_pct": 8.0,
                        "excess_vs_0050_60d_pct": 5.0,
                        "excess_vs_00631L_60d_pct": 2.0,
                        "rs20_vs_0050": 6.0,
                        "rs20_vs_00631L": 5.0,
                    },
                    {
                        "signal_date": "2024-01-04",
                        "next_tradable_date": "2024-01-05",
                        "ticker": "2330",
                        "candidate_name": "台積電",
                        "candidate_source": "dynamic_pool1",
                        "candidate_layer": "core",
                        "event_variant": "trend_ext_ma_stack_breakout",
                        "case_trace_only": False,
                        "uses_forward_return_as_rule": False,
                        "entry_price": 100.0,
                        "event_return_20d_pct": 7.0,
                        "event_return_40d_pct": 9.0,
                        "excess_vs_0050_60d_pct": 2.0,
                        "excess_vs_00631L_60d_pct": -2.0,
                        "rs20_vs_0050": 3.0,
                        "rs20_vs_00631L": 2.0,
                    },
                ]
            ).to_csv(outcome_dir / "trend_extension_exact_event_outcome_panel.csv", index=False)

            formal_dir = root / "outputs" / "combined_formal_target_stream_20150128_20211230_20260702"
            formal_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "execution_date": "2024-01-03",
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    },
                    {
                        "signal_date": "2024-01-03",
                        "execution_date": "2024-01-04",
                        "formal_target": "00631L.TW",
                        "target_type": "market_exposure",
                        "risk_off_state": "formal_target_active",
                    },
                    {
                        "signal_date": "2024-01-04",
                        "execution_date": "2024-01-05",
                        "formal_target": "2454.TW",
                        "target_type": "stock",
                        "risk_off_state": "formal_target_active",
                    },
                ]
            ).to_csv(formal_dir / "combined_formal_target_stream.csv", index=False)

            context_dir = root / "outputs" / "dynamic_pool1_candidate_panel_v0_20260704"
            context_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"ticker": "2330", "market": "TWSE"},
                    {"ticker": "2308", "market": "TWSE"},
                ]
            ).to_csv(context_dir / "candidate_pool_by_month.csv", index=False)

            manifest = run_strong_stock_trend_extension_event_to_action_contract(
                repo_root=root,
                outcome_panel=outcome_dir / "trend_extension_exact_event_outcome_panel.csv",
                candidate_context=context_dir / "candidate_pool_by_month.csv",
                output_dir=root / "contract",
            )

            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertEqual(manifest["formal_direct_stock_target_override_count"], 0)
            self.assertEqual(manifest["proxy_rows_in_action_contract"], 0)
            contract = pd.read_csv(root / "contract" / "trend_extension_event_to_action_contract.csv")
            allowed = contract[contract["action_allowed"].astype(bool)]
            self.assertEqual(set(allowed["ticker"]), {"2330.TW"})
            self.assertFalse(allowed["uses_forward_return_as_rule"].any())
            blocked = pd.read_csv(root / "contract" / "trend_extension_conflict_blocked_rows.csv")
            self.assertTrue(blocked["action_blocked_reason"].astype(str).str.contains("reference_only|direct_stock").any())
            caution = pd.read_csv(root / "contract" / "trend_extension_00631l_caution_audit.csv")
            self.assertFalse(caution["used_as_rule"].any())


if __name__ == "__main__":
    unittest.main()
