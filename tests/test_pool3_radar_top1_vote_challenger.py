from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_radar_top1_vote_challenger import run_pool3_radar_top1_vote_challenger


class Pool3RadarTop1VoteChallengerTest(unittest.TestCase):
    def test_runner_outputs_shadow_top1_vote_without_formal_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "weighted.csv"
            membership = root / "membership.csv"
            readiness = root / "readiness.json"
            output = root / "out"
            pd.DataFrame(
                [
                    _basket("2024-01-02", "3260.TWO", 0.12, 100, 0.01),
                    _basket("2024-01-02", "2344.TW", 0.10, 50, 0.01),
                    _basket("2024-01-03", "3260.TWO", 0.13, 101, 0.01),
                    _basket("2024-01-03", "2344.TW", 0.10, 51, 0.02),
                    _basket("2024-01-04", "3260.TWO", 0.14, 102, 0.01),
                    _basket("2024-01-04", "2344.TW", 0.10, 52, 0.02),
                ]
            ).to_csv(source, index=False)
            pd.DataFrame(
                [
                    {"ticker": "3260.TWO", "review_status": "accepted", "usable_for_formal_replay": True},
                    {"ticker": "2344.TW", "review_status": "accepted", "usable_for_formal_replay": True},
                ]
            ).to_csv(membership, index=False)
            readiness.write_text(
                json.dumps(
                    {
                        "theme_membership_v2_ready": True,
                        "coverage_ratio_after_batch": 1.0,
                        "future_data_violation_count": 0,
                        "remaining_gap_symbol_count": 0,
                        "ready_threshold": 0.95,
                    }
                ),
                encoding="utf-8",
            )

            result = run_pool3_radar_top1_vote_challenger(
                weighted_basket_daily=source,
                membership_csv=membership,
                readiness_manifest=readiness,
                output_dir=output,
                min_persistence_days=3,
            )

            panel = pd.read_csv(result / "pool3_radar_top1_vote_panel.csv")
            last = panel.iloc[-1]
            self.assertEqual(last["vote_target"], "3260.TWO")
            self.assertTrue(bool(last["challenger_eligible_for_pool_selection"]))
            self.assertFalse(bool(last["eligible_for_pool_selection"]))
            self.assertFalse(bool(last["active_in_trade_decision"]))
            self.assertFalse(bool(last["formal_model_changed"]))
            self.assertFalse(bool(last["pool3_formal_vote_changed"]))
            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pool3_radar_vote_mode"], "top1_shadow")
            self.assertFalse(manifest["active_in_trade_decision"])

    def test_unaccepted_or_overheated_top1_is_observation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "weighted.csv"
            membership = root / "membership.csv"
            readiness = root / "readiness.json"
            output = root / "out"
            pd.DataFrame(
                [
                    _basket("2024-01-02", "9999.TW", 0.12, 100, 0.30),
                    _basket("2024-01-03", "9999.TW", 0.13, 101, 0.01),
                    _basket("2024-01-04", "9999.TW", 0.14, 102, 0.01),
                ]
            ).to_csv(source, index=False)
            pd.DataFrame(
                [{"ticker": "3260.TWO", "review_status": "accepted", "usable_for_formal_replay": True}]
            ).to_csv(membership, index=False)
            readiness.write_text(
                json.dumps(
                    {
                        "theme_membership_v2_ready": True,
                        "coverage_ratio_after_batch": 1.0,
                        "future_data_violation_count": 0,
                        "remaining_gap_symbol_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_pool3_radar_top1_vote_challenger(
                weighted_basket_daily=source,
                membership_csv=membership,
                readiness_manifest=readiness,
                output_dir=output,
                min_persistence_days=1,
            )

            panel = pd.read_csv(result / "pool3_radar_top1_vote_panel.csv")
            self.assertTrue(panel["vote_target"].fillna("").eq("").all())
            self.assertIn("accepted_universe", ";".join(panel["pool3_radar_top1_ineligible_reason"].fillna("")))
            self.assertIn("overheat", ";".join(panel["pool3_radar_top1_ineligible_reason"].fillna("")))


def _basket(date: str, ticker: str, weight: float, close: float, ret: float) -> dict:
    return {
        "date": date,
        "period": "2024",
        "variant": "top10_base",
        "ticker": ticker,
        "theme": "記憶體",
        "weight": weight,
        "close": close,
        "return": ret,
        "basket_value": 1_000_000,
        "cash": 0,
        "market_exposure": 1,
        "trade_action": "",
        "trade_gross_amount": 0,
        "trade_cost": 0,
        "data_status": "actual_top10_replay",
    }


if __name__ == "__main__":
    unittest.main()
