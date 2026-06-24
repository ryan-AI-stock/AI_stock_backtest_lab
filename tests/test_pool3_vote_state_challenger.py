from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_vote_state_challenger import run_pool3_vote_state_challenger
from backtest_lab.stock_pool_consensus import build_consensus


class Pool3VoteStateChallengerTest(unittest.TestCase):
    def test_builds_vote_state_panels_without_etf_stock_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "2454.TW", True, "formal_candidate"),
                    _row(
                        "2024-01-02",
                        "large_core_bluechip_v0",
                        "00631L.TW",
                        True,
                        "market_exposure_tool",
                        top_asset_type="etf",
                    ),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=1, asset_type="etf"),
                    _top("2024-01-02", "2882.TW", "formal_candidate", True, rank=2),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_vote_state_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["trade_decision_changed"])
            self.assertFalse(metadata["active_in_trade_decision"])
            summary = pd.read_csv(output / "pool3_vote_state_variant_summary.csv")
            self.assertEqual(set(summary["pool3_etf_stock_vote_rows"]), {0})
            self.assertEqual(set(summary["pool3_etf_exact_consensus_rows"]), {0})
            panel = pd.read_csv(output / "pool3_vote_state_style_base_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["pool3_vote_state"], "full_stock_vote")
            self.assertEqual(pool3["top_ticker"], "2882.TW")
            self.assertTrue(_truthy(pool3["eligible_for_exact_ticker_consensus"]))

    def test_direction_support_only_counts_direction_but_not_exact_consensus(self) -> None:
        manifest = {
            "signal_date": "2024-01-02",
            "generated": [
                _manifest_item("pool1", "台積電(2330)", "2330.TW", "stock", "formal_candidate", True),
                _manifest_item("pool2", "0050正二(00631L)", "00631L.TW", "etf", "market_exposure_tool", True),
                _manifest_item("pool3", "國泰金(2882)", "2882.TW", "stock", "direction_support_only", False),
            ],
            "skipped": [],
        }

        consensus = build_consensus(manifest)

        health = consensus["health_diagnostic"]
        self.assertFalse(health["exact_ticker_consensus"])
        self.assertTrue(health["direction_consensus"])
        self.assertEqual(health["direction_consensus_group"], "stock_attack")
        self.assertEqual(consensus["result_state"], "divergent")


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
        "asset_type": asset_type,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
    }


def _manifest_item(
    pool_id: str,
    display: str,
    ticker: str,
    asset_type: str,
    selection_layer: str,
    eligible: bool,
) -> dict[str, object]:
    return {
        "pool_id": pool_id,
        "pool_name": pool_id,
        "vote_group": "three_perspective_v1",
        "top_display": display,
        "top_ticker": ticker,
        "top_asset_type": asset_type,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
        "selection_reason": "",
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
