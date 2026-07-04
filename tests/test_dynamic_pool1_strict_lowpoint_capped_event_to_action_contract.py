import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_strict_lowpoint_capped_event_to_action_contract import (
    run_dynamic_pool1_strict_lowpoint_capped_event_to_action_contract,
)


class DynamicPool1StrictLowpointCappedEventToActionContractTest(unittest.TestCase):
    def test_caps_aggregate_sleeve_and_blocks_pyramiding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "event_contract"
            source.mkdir()
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
                        "signal_date": "2024-01-04",
                        "next_tradable_date": "2024-01-05",
                        "ticker": "2308.TW",
                        "candidate_name": "台達電",
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
                        "signal_date": "2024-01-04",
                        "execution_date": "2024-01-05",
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    },
                ]
            ).to_csv(formal_dir / "combined_formal_target_stream.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            dates = pd.date_range("2024-01-02", periods=45, freq="B")
            rows = []
            for idx, date in enumerate(dates):
                for ticker, close, turnover in [("6669", 100, 1_000_000), ("2308", 50, 500_000)]:
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "market": "TWSE",
                            "close": close + idx,
                            "turnover": turnover + idx,
                        }
                    )
            pd.DataFrame(rows).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)

            manifest = run_dynamic_pool1_strict_lowpoint_capped_event_to_action_contract(
                repo_root=root,
                event_contract_dir=source,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )
            self.assertEqual(manifest["cap_violation_count"], 0)
            self.assertEqual(manifest["formal_direct_stock_target_override_count"], 0)
            contract = pd.read_csv(root / "out" / "strict_lowpoint_capped_event_to_action_contract.csv")
            self.assertFalse(contract["cap_violation"].any())
            grouped = contract.groupby("variant")["aggregate_sleeve_exposure"].max()
            caps = contract.groupby("variant")["max_sleeve_cap"].max()
            self.assertTrue((grouped <= caps).all())
            self.assertTrue(contract["blocked_by_active_sleeve"].any())
            blocked = pd.read_csv(root / "out" / "strict_lowpoint_blocked_by_active_sleeve.csv")
            self.assertFalse(blocked.empty)


if __name__ == "__main__":
    unittest.main()
