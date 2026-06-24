from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_pure_stock_low_correlation_challenger import (
    run_pool3_pure_stock_low_correlation_challenger,
)


class Pool3PureStockLowCorrelationChallengerTest(unittest.TestCase):
    def test_etf_is_not_pool3_stock_vote_or_exact_consensus(self) -> None:
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
                ]
            ).to_csv(top, index=False)

            output = run_pool3_pure_stock_low_correlation_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["trade_decision_changed"])
            self.assertFalse(metadata["active_in_trade_decision"])
            panel = pd.read_csv(output / "pool3_stock_only_plus_market_exposure_diagnostic_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["top_ticker"], "00631L.TW")
            self.assertEqual(pool3["selection_layer"], "market_exposure_tool")
            self.assertFalse(_truthy(pool3["eligible_for_pool_selection"]))
            self.assertFalse(_truthy(pool3["eligible_for_pool3_stock_vote"]))
            self.assertFalse(_truthy(pool3["eligible_for_exact_ticker_consensus"]))
            self.assertTrue(_truthy(pool3["eligible_for_market_exposure"]))

    def test_stock_only_base_can_replace_pool3_etf_with_first_stock_candidate(self) -> None:
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

            output = run_pool3_pure_stock_low_correlation_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "pool3_stock_only_style_base_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["top_ticker"], "2882.TW")
            self.assertEqual(pool3["selection_layer"], "formal_candidate")
            self.assertTrue(_truthy(pool3["eligible_for_pool_selection"]))
            self.assertTrue(_truthy(pool3["eligible_for_pool3_stock_vote"]))
            self.assertTrue(_truthy(pool3["eligible_for_exact_ticker_consensus"]))

    def test_low_correlation_variant_skips_same_ticker_as_peer_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2882.TW", True, "formal_candidate"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "2454.TW", True, "formal_candidate"),
                    _row("2024-01-02", "large_core_bluechip_v0", "2882.TW", True, "formal_candidate"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "2882.TW", "formal_candidate", True, rank=1),
                    _top("2024-01-02", "2891.TW", "formal_candidate", True, rank=2),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_pure_stock_low_correlation_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "pool3_stock_only_low_correlation_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["top_ticker"], "2891.TW")
            self.assertTrue(_truthy(pool3["eligible_for_pool3_stock_vote"]))
            self.assertEqual(pool3["correlation_to_pool1_signal"], "different_ticker")


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
        "gate_rule_id": "core_style_complement_opportunity_gate_v1",
        "gate_reason": "",
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


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
