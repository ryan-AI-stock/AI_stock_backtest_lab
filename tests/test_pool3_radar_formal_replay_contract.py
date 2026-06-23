from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_radar_formal_replay_contract import run_pool3_radar_formal_replay_contract


class Pool3RadarFormalReplayContractTest(unittest.TestCase):
    def test_contract_rejects_proxy_baseline_and_blocked_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.csv"
            overlay = root / "overlay.csv"
            readiness = root / "readiness.json"
            output = root / "out"
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "pool1_vote": "00631L.TW",
                        "pool2_vote": "2327.TW",
                        "pool3_vote": "00631L.TW",
                        "consensus_state": "consensus",
                        "winner_ticker": "00631L.TW",
                        "position_ticker": "00631L.TW",
                        "cash": 0,
                        "equity": 1_000_000,
                        "drawdown": 0,
                        "turnover": 0,
                        "transaction_cost": 0,
                        "data_status": "partial_proxy_from_stride20_vote_panel",
                    }
                ]
            ).to_csv(baseline, index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "variant": "ma200_radar20_00631l80_else_top10",
                        "pool3_formal_vote": "weighted_basket",
                        "holding_ticker": "00631L.TW",
                        "holding_name": "0050正二",
                        "theme": "market_exposure",
                        "weight": 0.8,
                        "shares": 1000,
                        "fill_action": "hold",
                        "fill_price": 30,
                        "cash": 0,
                        "position_value": 800_000,
                        "transaction_cost": 0,
                        "equity": 1_000_000,
                    }
                ]
            ).to_csv(overlay, index=False)
            readiness.write_text(
                json.dumps(
                    {
                        "can_core_absorb_as_formal_challenger": False,
                        "blockers": ["formal_top3_ready=false"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_pool3_radar_formal_replay_contract(
                baseline_daily=baseline,
                overlay_daily=overlay,
                readiness_manifest=readiness,
                output_dir=output,
            )

            manifest = json.loads((result / "formal_replay_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "contract_ready_inputs_pending")
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["formal_model_changed"])
            failed = {row["check_id"] for row in manifest["failed_checks"]}
            self.assertIn("baseline_three_pool_formal_daily:not_proxy", failed)
            self.assertIn("radar_readiness:formal_absorb_flag", failed)
            self.assertIn("radar_readiness:no_blockers", failed)
            self.assertTrue((result / "required_baseline_three_pool_formal_daily_schema.csv").exists())
            self.assertTrue((result / "required_pool3_radar_weighted_overlay_formal_daily_schema.csv").exists())
            self.assertTrue((result / "required_formal_decision_diff_schema.csv").exists())

    def test_contract_accepts_schema_clean_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.csv"
            overlay = root / "overlay.csv"
            readiness = root / "readiness.json"
            output = root / "out"
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "pool1_vote": "00631L.TW",
                        "pool2_vote": "2327.TW",
                        "pool3_vote": "00631L.TW",
                        "consensus_state": "consensus",
                        "winner_ticker": "00631L.TW",
                        "position_ticker": "00631L.TW",
                        "cash": 0,
                        "equity": 1_000_000,
                        "drawdown": 0,
                        "turnover": 0,
                        "transaction_cost": 0,
                    }
                ]
            ).to_csv(baseline, index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "variant": "ma200_radar20_00631l80_else_top10",
                        "pool3_formal_vote": "weighted_basket",
                        "holding_ticker": "3260.TW",
                        "holding_name": "威剛",
                        "theme": "記憶體",
                        "weight": 0.2,
                        "shares": 100,
                        "fill_action": "buy",
                        "fill_price": 100,
                        "cash": 800_000,
                        "position_value": 20_000,
                        "transaction_cost": 50,
                        "equity": 1_000_000,
                    }
                ]
            ).to_csv(overlay, index=False)
            readiness.write_text(
                json.dumps(
                    {
                        "can_core_absorb_as_formal_challenger": True,
                        "blockers": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = run_pool3_radar_formal_replay_contract(
                baseline_daily=baseline,
                overlay_daily=overlay,
                readiness_manifest=readiness,
                output_dir=output,
            )

            manifest = json.loads((result / "formal_replay_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "inputs_pass_contract")
            self.assertTrue(manifest["accepted_for_formal_challenger_replay"])
            self.assertFalse(manifest["active_in_trade_decision"])


if __name__ == "__main__":
    unittest.main()
