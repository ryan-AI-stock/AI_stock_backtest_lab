import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.date_aware_dynamic_universe_state_replay import run_date_aware_dynamic_universe_state_replay


class DateAwareDynamicUniverseStateReplayTest(unittest.TestCase):
    def test_builds_dynamic_universe_coverage_but_blocks_formal_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel"
            lifecycle = root / "lifecycle"
            output = root / "out"
            panel.mkdir()
            lifecycle.mkdir()

            (panel / "manifest.json").write_text(
                json.dumps({"date_start": "2014-11-03", "date_end": "2015-01-28"}, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame([{"date": "2014-11-03"}, {"date": "2015-01-28"}]).to_csv(
                panel / "formal_policy_input_readiness.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"date": "2015-01-28", "candidate_ticker": "2330.TW", "candidate_name": "台積電", "score": 1.1, "rank": 1}
                ]
            ).to_csv(panel / "pool1_daily_candidate_ranking_panel.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2014-11-03", "ticker": "2330.TW", "candidate_available_for_pool1_ranking": False},
                    {"date": "2015-01-28", "ticker": "2330.TW", "candidate_available_for_pool1_ranking": True},
                    {"date": "2015-01-28", "ticker": "2454.TW", "candidate_available_for_pool1_ranking": True},
                ]
            ).to_csv(lifecycle / "pool1_date_aware_candidate_availability_daily.csv", index=False)
            pd.DataFrame(
                [
                    {"ticker": "2330.TW", "first_pool1_scoring_date": "2015-01-28"},
                    {"ticker": "2454.TW", "first_pool1_scoring_date": "2015-01-28"},
                ]
            ).to_csv(lifecycle / "pool1_ticker_lifecycle_contract.csv", index=False)

            result = run_date_aware_dynamic_universe_state_replay(panel_dir=panel, lifecycle_dir=lifecycle, output_dir=output)
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["dynamic_universe_contract_defined"])
            self.assertTrue(manifest["daily_dynamic_candidate_universe_ready"])
            self.assertFalse(manifest["dynamic_universe_state_replay_formal_ready"])
            self.assertFalse(manifest["no_target_cash_all_applied"])

            coverage = pd.read_csv(output / "dynamic_universe_state_replay_coverage.csv")
            first = coverage[coverage["signal_date"].eq("2014-11-03")].iloc[0]
            self.assertEqual(first["available_universe_count"], 0)
            second = coverage[coverage["signal_date"].eq("2015-01-28")].iloc[0]
            self.assertEqual(second["available_universe_count"], 2)
            self.assertEqual(second["pool1_top_candidate"], "2330.TW")
            self.assertFalse(bool(second["source_formal_ready"]))

            blocked = pd.read_csv(output / "blocked_signal_rows.csv")
            self.assertFalse(blocked["source_formal_ready"].any())
            self.assertFalse(blocked["no_target_cash_all_applied"].any())

            decisions = pd.read_csv(output / "proxy_or_formal_source_decision.csv")
            rejected = decisions[decisions["source_layer"].eq("ranking_first_as_formal_target")].iloc[0]
            self.assertEqual(rejected["status"], "rejected")

            blockers = pd.read_csv(output / "blocker_by_field.csv")
            self.assertIn("missing_daily_fallback_score_margin_panel", set(blockers["blocker"]))


if __name__ == "__main__":
    unittest.main()
