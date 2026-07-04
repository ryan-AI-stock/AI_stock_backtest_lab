import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_strict_lowpoint_event_contract import run_dynamic_pool1_strict_lowpoint_event_contract


class DynamicPool1StrictLowpointEventContractTest(unittest.TestCase):
    def test_builds_primary_and_reference_event_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "ticker": "6669.TW",
                        "candidate_name": "緯穎",
                        "strict_variant_id": "lowpoint_0_2d_rebound_5_12pct",
                        "candidate_layer": "core",
                        "layer_group": "ai",
                        "days_since_10d_low": 1,
                        "rebound_from_10d_low_pct": 6,
                        "days_since_low_band": "0_2d",
                        "rebound_from_low_band": "5_12pct",
                        "close": 100,
                        "ma20": 95,
                        "ma60": 90,
                        "ma120": 80,
                        "close_vs_ma20_pct": 5,
                        "close_vs_ma60_pct": 10,
                        "drawdown_from_20d_high_pct": -3,
                        "drawdown_from_60d_high_pct": -5,
                        "rs_vs_0050_5d_pct": 2,
                        "rs_vs_00631L_5d_pct": 3,
                        "rs60_positive_vs_both_at_event": True,
                    },
                    {
                        "signal_date": "2024-01-03",
                        "ticker": "2308.TW",
                        "candidate_name": "台達電",
                        "strict_variant_id": "lowpoint_3_5d_rebound_5_12pct",
                        "candidate_layer": "watch",
                        "layer_group": "other",
                        "days_since_10d_low": 4,
                        "rebound_from_10d_low_pct": 7,
                        "days_since_low_band": "3_5d",
                        "rebound_from_low_band": "5_12pct",
                        "close": 100,
                        "ma20": 95,
                        "ma60": 90,
                        "ma120": 80,
                        "close_vs_ma20_pct": 5,
                        "close_vs_ma60_pct": 10,
                        "drawdown_from_20d_high_pct": -3,
                        "drawdown_from_60d_high_pct": -5,
                        "rs_vs_0050_5d_pct": 2,
                        "rs_vs_00631L_5d_pct": 3,
                        "rs60_positive_vs_both_at_event": True,
                    },
                ]
            ).to_csv(source / "strict_lowpoint_timing_band_event_panel.csv", index=False)
            pd.DataFrame([{"variant": "negative_control"}]).to_csv(source / "negative_control_summary.csv", index=False)
            liquidity = root / "liquidity" / "shards"
            liquidity.mkdir(parents=True)
            pd.DataFrame([{"date": "2024-01-03"}, {"date": "2024-01-04"}]).to_csv(
                liquidity / "accepted_liquidity_rows_2024_01.csv", index=False
            )
            manifest = run_dynamic_pool1_strict_lowpoint_event_contract(
                source_dir=source,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )
            self.assertEqual(manifest["future_data_violation_count"], 0)
            contract = pd.read_csv(root / "out" / "strict_lowpoint_event_contract.csv")
            self.assertIn("strict_lowpoint_0_2d_rebound_5_12pct", set(contract["event_variant"]))
            ref = contract[contract["event_variant"].str.contains("reference_only")]
            self.assertFalse(ref.empty)
            self.assertTrue((ref["event_variant_role"] == "reference_only").all())
            self.assertFalse(contract["uses_forward_return_as_rule"].any())
            case_trace = pd.read_csv(root / "out" / "case_trace_6669_2308_2317.csv")
            self.assertIn("2317.TW", set(case_trace["ticker"].astype(str)))
            foxconn = case_trace[case_trace["ticker"].astype(str).eq("2317.TW")].iloc[0]
            self.assertFalse(bool(foxconn["event_found"]))
            self.assertEqual(foxconn["case_trace_blocked_reason"], "no_strict_lowpoint_event_for_case_ticker")
            self.assertTrue(manifest["case_trace_contains_2317"])


if __name__ == "__main__":
    unittest.main()
