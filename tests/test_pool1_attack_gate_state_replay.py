import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_attack_gate_state_replay import run_pool1_attack_gate_state_replay


class Pool1AttackGateStateReplayTest(unittest.TestCase):
    def test_blocks_state_replay_until_dynamic_universe_adapter_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel"
            lifecycle = root / "lifecycle"
            output = root / "out"
            panel.mkdir()
            lifecycle.mkdir()

            (panel / "manifest.json").write_text(
                json.dumps({"date_start": "2014-11-03", "date_end": "2018-02-06"}, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"date": "2014-11-03"},
                    {"date": "2018-02-05"},
                    {"date": "2018-02-06"},
                ]
            ).to_csv(panel / "formal_policy_input_readiness.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2018-02-06", "candidate_ticker": "6669.TW", "candidate_name": "緯穎", "score": 1.2, "rank": 1}
                ]
            ).to_csv(panel / "pool1_daily_candidate_ranking_panel.csv", index=False)
            pd.DataFrame(
                [
                    {"ticker": "2330.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "2454.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "2308.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "2317.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "2382.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "3231.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "00631L.TW", "first_pool1_scoring_date": "2015-01-27"},
                    {"ticker": "6669.TW", "first_pool1_scoring_date": "2018-02-06"},
                ]
            ).to_csv(lifecycle / "pool1_ticker_lifecycle_contract.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2014-11-03",
                        "ticker": "6669.TW",
                        "candidate_available_for_pool1_ranking": False,
                    },
                    {
                        "date": "2018-02-06",
                        "ticker": "6669.TW",
                        "candidate_available_for_pool1_ranking": True,
                    },
                ]
            ).to_csv(lifecycle / "pool1_date_aware_candidate_availability_daily.csv", index=False)

            result = run_pool1_attack_gate_state_replay(panel_dir=panel, lifecycle_dir=lifecycle, output_dir=output)
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["pool1_attack_gate_state_formal_ready"])
            self.assertTrue(manifest["dynamic_universe_state_replay_required"])
            self.assertEqual(manifest["static_all_ticker_scoring_ready_date"], "2018-02-06")
            self.assertFalse(manifest["no_target_cash_all_applied"])

            blocked = pd.read_csv(output / "blocked_signal_rows.csv")
            self.assertEqual(len(blocked), 3)
            self.assertFalse(blocked["source_formal_ready"].any())
            self.assertFalse(blocked["no_target_cash_all_applied"].any())

            decisions = pd.read_csv(output / "proxy_or_formal_source_decision.csv")
            rejected = decisions[decisions["source_layer"].eq("ranking_first_proxy")].iloc[0]
            self.assertEqual(rejected["status"], "rejected")

            blockers = pd.read_csv(output / "blocker_by_field.csv")
            self.assertIn("static_common_date_universe_blocks_2014_start", set(blockers["blocker"]))


if __name__ == "__main__":
    unittest.main()
