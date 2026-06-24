from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_style_complement_v2_challenger import run_pool3_style_complement_v2_challenger


class Pool3StyleComplementV2ChallengerTest(unittest.TestCase):
    def test_builds_v2_panels_without_changing_formal_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "2330.TW", True, "formal_candidate"),
                    _row(
                        "2024-01-02",
                        "large_core_bluechip_v0",
                        "2882.TW",
                        True,
                        "formal_candidate",
                        top_asset_type="stock",
                        gate_reason="風格補強池 v1：20日回撤控管=-9.0%(Y)",
                    ),
                    _row("2024-01-03", "ai_theme_large_cap_v20260613", "00631L.TW", True, "market_exposure_tool"),
                    _row("2024-01-03", "tw50_dynamic_constituents_v0", "00631L.TW", True, "market_exposure_tool"),
                    _row(
                        "2024-01-03",
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
                    _top("2024-01-02", "2882.TW", "formal_candidate", True, rank=1),
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=2),
                    _top("2024-01-03", "00631L.TW", "market_exposure_tool", True, rank=1),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_style_complement_v2_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["trade_decision_changed"])
            self.assertFalse(metadata["active_in_trade_decision"])
            diff = pd.read_csv(output / "pool3_style_complement_v2_decision_diff.csv")
            self.assertIn("style_complement_v2_mdd_cap", set(diff["variant"]))
            mdd_panel = pd.read_csv(output / "style_complement_v2_mdd_cap_replay_panel.csv")
            pool3 = mdd_panel[(mdd_panel["pool_id"] == "large_core_bluechip_v0") & (mdd_panel["requested_signal_date"] == "2024-01-02")].iloc[0]
            self.assertFalse(_truthy(pool3["eligible_for_pool_selection"]))
            self.assertEqual(pool3["selection_layer"], "observation_only")
            self.assertEqual(pool3["gate_rule_id"], "core_style_complement_opportunity_gate_v2")

    def test_combined_fallback_replaces_blocked_stock_with_market_exposure_tool(self) -> None:
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
                        "2882.TW",
                        True,
                        "formal_candidate",
                        top_asset_type="stock",
                        gate_reason="風格補強池 v1：20日回撤控管=-9.0%(Y)",
                    ),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "2882.TW", "formal_candidate", True, rank=1),
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=2),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_style_complement_v2_challenger(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "style_complement_v2_mdd_cap_plus_consensus_aware_plus_fallback_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["top_ticker"], "00631L.TW")
            self.assertEqual(pool3["selection_layer"], "market_exposure_tool")
            self.assertTrue(_truthy(pool3["eligible_for_pool_selection"]))
            self.assertEqual(pool3["pool3_fallback_state"], "market_exposure_tool")


def _row(
    date: str,
    pool_id: str,
    ticker: str,
    eligible: bool,
    selection_layer: str,
    *,
    top_asset_type: str = "stock",
    gate_reason: str = "",
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
        "gate_reason": gate_reason,
    }


def _top(date: str, ticker: str, selection_layer: str, eligible: bool, *, rank: int) -> dict[str, object]:
    return {
        "period": "2024",
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": "large_core_bluechip_v0",
        "ticker": ticker,
        "display": ticker,
        "rank": rank,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
