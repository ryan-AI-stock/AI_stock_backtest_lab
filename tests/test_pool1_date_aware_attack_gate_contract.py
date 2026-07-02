import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_date_aware_attack_gate_contract import run_pool1_date_aware_attack_gate_contract


class Pool1DateAwareAttackGateContractTest(unittest.TestCase):
    def test_builds_blocked_contract_without_applying_no_target_cash_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel"
            output = root / "out"
            panel.mkdir()

            (panel / "manifest.json").write_text(
                json.dumps(
                    {
                        "date_start": "2014-11-03",
                        "date_end": "2014-11-04",
                        "trading_date_count": 2,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "date": "2014-11-03",
                        "pool_id": "ai_theme_strategy",
                        "pool_name": "AI主線池",
                        "candidate_ticker": "2454.TW",
                        "candidate_name": "聯發科",
                        "score": 1.23,
                        "raw_rank": 1,
                        "rank": 1,
                        "passed": True,
                        "attack_gate_status": "ranking_reconstructed_attack_gate_not_reconstructed",
                        "reason": "ranking only",
                        "formal_vote_ready": False,
                        "price_only_used": True,
                        "adjusted_close_available": True,
                    },
                    {
                        "date": "2014-11-04",
                        "pool_id": "ai_theme_strategy",
                        "pool_name": "AI主線池",
                        "candidate_ticker": "00631L.TW",
                        "candidate_name": "0050正二",
                        "score": 0.5,
                        "raw_rank": 1,
                        "rank": 1,
                        "passed": True,
                        "attack_gate_status": "ranking_reconstructed_attack_gate_not_reconstructed",
                        "reason": "ranking only",
                        "formal_vote_ready": False,
                        "price_only_used": True,
                        "adjusted_close_available": True,
                    },
                ]
            ).to_csv(panel / "pool1_daily_candidate_ranking_panel.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2014-11-03",
                        "pool1_top_candidate": "2454.TW",
                        "pool1_formal_vote_ready": False,
                        "readiness_state": "blocked_for_formal_target_stream",
                        "blocker_reason": "missing_date_aware_pool1_attack_gate_and_formal_target_contract",
                    },
                    {
                        "date": "2014-11-04",
                        "pool1_top_candidate": "00631L.TW",
                        "pool1_formal_vote_ready": False,
                        "readiness_state": "blocked_for_formal_target_stream",
                        "blocker_reason": "missing_date_aware_pool1_attack_gate_and_formal_target_contract",
                    },
                ]
            ).to_csv(panel / "formal_policy_input_readiness.csv", index=False)

            result = run_pool1_date_aware_attack_gate_contract(panel_dir=panel, output_dir=output)
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["pool1_attack_gate_formal_ready"])
            self.assertFalse(manifest["no_target_cash_all_applied"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])

            contract = pd.read_csv(output / "pool1_attack_gate_contract.csv")
            self.assertIn("attack_gate_active", set(contract["field_name"]))
            self.assertIn("target_is_actionable", set(contract["field_name"]))
            self.assertIn("candidate_listed_and_tradable_on_signal_date", set(contract["field_name"]))

            blocked = pd.read_csv(output / "blocked_signal_rows.csv")
            self.assertEqual(len(blocked), 2)
            self.assertTrue(blocked["formal_target"].fillna("").eq("").all())
            self.assertTrue(blocked["risk_off_state"].eq("not_evaluated_blocked").all())
            self.assertFalse(blocked["no_target_cash_all_applied"].any())

            blockers = pd.read_csv(output / "blocker_by_field.csv")
            self.assertIn("missing_date_aware_pool1_attack_gate_state", set(blockers["blocker"]))


if __name__ == "__main__":
    unittest.main()
