from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_direction_state_challenger import run_pool3_direction_state_challenger


class Pool3DirectionStateChallengerTest(unittest.TestCase):
    def test_direction_state_runner_keeps_etf_out_of_pool3_stock_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "00631L.TW", True, "market_exposure_tool", top_asset_type="etf"),
                    _row("2024-01-02", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool", top_asset_type="etf"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=1, asset_type="etf"),
                    _top("2024-01-02", "2603.TW", "formal_candidate", True, rank=2, score=0.72),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_direction_state_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["trade_decision_changed"])
            summary = pd.read_csv(output / "pool3_direction_state_variant_summary.csv")
            self.assertEqual(set(summary["pool3_etf_stock_vote_rows"]), {0})
            self.assertEqual(set(summary["pool3_etf_exact_consensus_rows"]), {0})
            panel = pd.read_csv(output / "pool3_direction_state_base_v1_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["pool3_direction_state"], "attack_confirmed")
            self.assertEqual(pool3["pool3_vote_state"], "full_stock_vote")
            self.assertEqual(pool3["top_ticker"], "2603.TW")

    def test_attack_candidate_can_be_direction_support_without_exact_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "2454.TW", True, "formal_candidate"),
                    _row("2024-01-02", "large_core_bluechip_v0", "2603.TW", False, "observation_only"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "2603.TW", "formal_candidate", True, rank=4, score=0.31),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_direction_state_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "pool3_direction_state_attack_candidate_v1_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["pool3_direction_state"], "attack_candidate")
            self.assertEqual(pool3["pool3_vote_state"], "direction_support_only")
            self.assertFalse(_truthy(pool3["eligible_for_pool_selection"]))
            self.assertFalse(_truthy(pool3["eligible_for_exact_ticker_consensus"]))


def _row(
    date: str,
    pool_id: str,
    ticker: str,
    eligible: bool,
    selection_layer: str,
    *,
    top_asset_type: str = "stock",
) -> dict[str, object]:
    return {
        "period": "2024",
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": pool_id,
        "pool_name": pool_id,
        "top_ticker": ticker,
        "top_display": ticker,
        "top_asset_type": top_asset_type,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
        "gate_rule_id": "",
        "gate_reason": "",
        "status": "generated",
    }


def _top(
    date: str,
    ticker: str,
    selection_layer: str,
    eligible: bool,
    *,
    rank: int,
    score: float = 0.0,
    asset_type: str = "stock",
) -> dict[str, object]:
    return {
        "period": "2024",
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": "large_core_bluechip_v0",
        "ticker": ticker,
        "display": ticker,
        "rank": rank,
        "score": score,
        "asset_type": asset_type,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
