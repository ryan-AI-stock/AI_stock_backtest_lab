import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.formal_long_range_signal_reconstruction import run_formal_long_range_signal_reconstruction


class FormalLongRangeSignalReconstructionTest(unittest.TestCase):
    def test_builds_partial_and_formal_ready_target_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel2014"
            replay = root / "replay"
            pit = root / "pit"
            price = root / "price"
            output = root / "out"
            panel.mkdir()
            replay.mkdir()
            pit.mkdir()
            price.mkdir()

            (panel / "manifest.json").write_text(
                json.dumps(
                    {
                        "date_start": "2014-11-03",
                        "date_end": "2014-11-04",
                        "pool1_daily_candidate_ranking_panel_generated": True,
                        "pool2_daily_confirmation_panel_generated": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "date": "2014-11-03",
                        "pool1_top_candidate": "",
                        "pool1_formal_vote_ready": False,
                        "pool2_vote": "",
                        "pool2_confirmation_ready": False,
                        "anchor_after_query_date": True,
                        "sufficient_for_pool1_primary_pool2_confirmation": False,
                        "readiness_state": "blocked_for_formal_target_stream",
                        "blocker_reason": "missing_date_aware_pool1_attack_gate",
                    }
                ]
            ).to_csv(panel / "formal_policy_input_readiness.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "blocker": "pool1_date_aware_formal_attack_gate_contract",
                        "status": "missing",
                        "blocks_formal_target_stream": True,
                        "detail": "missing contract",
                        "next_owner": "Core",
                    }
                ]
            ).to_csv(panel / "data_blockers.csv", index=False)

            pd.DataFrame(
                [
                    {"period": "2022", "date": "2022-01-03", "pool1_vote": "2454.TW", "pool2_vote": "2454.TW"},
                    {"period": "2022", "date": "2022-01-04", "pool1_vote": "", "pool2_vote": "2454.TW"},
                    {"period": "2022", "date": "2022-01-05", "pool1_vote": "00631L.TW", "pool2_vote": "2327.TW"},
                ]
            ).to_csv(replay / "formal_three_pool_decision_panel.csv", index=False)

            result = run_formal_long_range_signal_reconstruction(
                panel_2014_dir=panel,
                formal_replay_dir=replay,
                pit_readiness_dir=pit,
                price_absorption_dir=price,
                output_dir=output,
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["2014_2021_formal_target_stream_ready"])
            self.assertTrue(manifest["2022_latest_formal_target_stream_ready"])
            self.assertEqual(manifest["no_target_risk_off_policy"], "cash_all")

            partial = pd.read_csv(output / "partial_target_stream.csv")
            blocked = partial[partial["signal_date"].astype(str).eq("2014-11-03")].iloc[0]
            self.assertEqual(blocked["risk_off_state"], "not_evaluated_blocked")
            self.assertFalse(bool(blocked["next_day_tradable_flag"]))

            formal = pd.read_csv(output / "formal_long_range_target_stream.csv")
            self.assertIn("no_target_cash_all", set(formal["risk_off_state"]))
            cash = formal[formal["risk_off_state"].eq("no_target_cash_all")].iloc[0]
            self.assertEqual(cash["formal_target"], "CASH")
            self.assertEqual(cash["formal_target_display"], "風險控管空手 / 現金")

            blockers = pd.read_csv(output / "blocked_periods.csv")
            self.assertIn("pool1_date_aware_formal_attack_gate_contract", set(blockers["blocker"]))


if __name__ == "__main__":
    unittest.main()
