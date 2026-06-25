from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_event_level_decision_diff import run_pool3_event_level_decision_diff


class Pool3EventLevelDecisionDiffTest(unittest.TestCase):
    def test_exact_consensus_missing_when_three_stock_votes_diverge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            formal = root / "formal.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "2454.TW", True),
                    _row("2024-01-02", "large_core_bluechip_v0", "2882.TW", True),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame([{"date": "2024-01-02", "consensus_state": "divergent", "winner_ticker": "", "action": "hold"}]).to_csv(
                formal,
                index=False,
            )

            output = run_pool3_event_level_decision_diff(
                replay_panel_path=replay,
                formal_decision_panel_path=formal,
                price_cache_dir=root,
                output_dir=root / "out",
            )

            event = pd.read_csv(output / "event_decision_diff_panel.csv").iloc[0]
            self.assertTrue(_truthy(event["pool3_has_full_stock_vote"]))
            self.assertEqual(event["exact_ticker_consensus"], "divergent")
            self.assertEqual(event["pool3_blocker_category"], "exact_consensus_missing")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])

    def test_pool2_no_vote_blocks_pool3_exact_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "", False, selection_layer="no_selection"),
                    _row("2024-01-02", "large_core_bluechip_v0", "2882.TW", True),
                ]
            ).to_csv(replay, index=False)

            output = run_pool3_event_level_decision_diff(
                replay_panel_path=replay,
                formal_decision_panel_path=None,
                price_cache_dir=root,
                output_dir=root / "out",
            )

            event = pd.read_csv(output / "event_decision_diff_panel.csv").iloc[0]
            self.assertEqual(event["pool2_vote_state"], "no_selection")
            self.assertEqual(event["pool3_blocker_category"], "pool2_no_vote_or_risk_off")
            summary = pd.read_csv(output / "pool3_vote_blocker_summary.csv")
            self.assertIn("pool2_no_vote_or_risk_off", set(summary["pool3_blocker_category"]))


def _row(
    date: str,
    pool_id: str,
    ticker: str,
    eligible: bool,
    *,
    selection_layer: str = "formal_candidate",
) -> dict[str, object]:
    return {
        "period": "2024",
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": pool_id,
        "pool_name": pool_id,
        "top_ticker": ticker,
        "top_display": ticker,
        "top_asset_type": "stock",
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
        "gate_rule_id": "fixture",
        "gate_reason": "",
        "blocked_reason": "" if eligible else "fixture blocked",
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
