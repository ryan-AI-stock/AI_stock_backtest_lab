import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_strict_lowpoint_event_to_action_contract import (
    run_dynamic_pool1_strict_lowpoint_event_to_action_contract,
)


class DynamicPool1StrictLowpointEventToActionContractTest(unittest.TestCase):
    def test_builds_bounded_actions_and_blocks_direct_stock_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "event_contract"
            source.mkdir()
            dates = pd.date_range("2024-01-02", periods=45, freq="B")
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "next_tradable_date": "2024-01-03",
                        "ticker": "6669.TW",
                        "candidate_name": "緯穎",
                        "event_variant": "strict_lowpoint_0_2d_rebound_5_12pct",
                        "event_variant_role": "primary",
                        "rs_vs_0050_3d_or_5d": 1.0,
                        "rs_vs_00631l_3d_or_5d": 1.0,
                    },
                    {
                        "signal_date": "2024-01-03",
                        "next_tradable_date": "2024-01-04",
                        "ticker": "2308.TW",
                        "candidate_name": "台達電",
                        "event_variant": "strict_lowpoint_3_5d_rebound_5_12pct_reference_only",
                        "event_variant_role": "reference_only",
                        "rs_vs_0050_3d_or_5d": 5.0,
                        "rs_vs_00631l_3d_or_5d": 5.0,
                    },
                    {
                        "signal_date": "2024-01-04",
                        "next_tradable_date": "2024-01-05",
                        "ticker": "6669.TW",
                        "candidate_name": "緯穎",
                        "event_variant": "strict_lowpoint_0_5d_rebound_5_12pct_short_rs_repair",
                        "event_variant_role": "primary",
                        "rs_vs_0050_3d_or_5d": 2.0,
                        "rs_vs_00631l_3d_or_5d": 2.0,
                    },
                ]
            ).to_csv(source / "strict_lowpoint_event_contract.csv", index=False)

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
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    },
                    {
                        "signal_date": "2024-01-04",
                        "execution_date": "2024-01-05",
                        "formal_target": "2330.TW",
                        "target_type": "stock",
                        "risk_off_state": "formal_target_active",
                    },
                ]
            ).to_csv(formal_dir / "combined_formal_target_stream.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            rows = []
            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": "6669",
                        "market": "TWSE",
                        "close": 100 + idx,
                        "turnover": 1_000_000 + idx,
                    }
                )
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": "2308",
                        "market": "TWSE",
                        "close": 50 + idx,
                        "turnover": 500_000 + idx,
                    }
                )
            pd.DataFrame(rows).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)

            manifest = run_dynamic_pool1_strict_lowpoint_event_to_action_contract(
                repo_root=root,
                event_contract_dir=source,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )
            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertEqual(manifest["formal_direct_stock_target_override_count"], 0)
            contract = pd.read_csv(root / "out" / "strict_lowpoint_event_to_action_contract.csv")
            allowed = contract[contract["action_allowed"].astype(bool)]
            self.assertEqual(set(allowed["ticker"]), {"6669.TW"})
            self.assertTrue((allowed["formal_state"] == "no_target").all())
            blocked = pd.read_csv(root / "out" / "strict_lowpoint_conflict_blocked_rows.csv")
            self.assertTrue(blocked["action_blocked_reason"].astype(str).str.contains("direct_stock|reference|context").any())
            case_trace = pd.read_csv(root / "out" / "strict_lowpoint_case_trace_6669_2308_2317.csv")
            self.assertIn("2317.TW", set(case_trace["ticker"].astype(str)))
            self.assertFalse(contract["uses_forward_return_as_rule"].any())


if __name__ == "__main__":
    unittest.main()
