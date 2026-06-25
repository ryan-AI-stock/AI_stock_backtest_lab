from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_independent_stock_expert import run_pool3_independent_stock_expert


class Pool3IndependentStockExpertTest(unittest.TestCase):
    def test_base_variant_votes_stock_without_core_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "ai_theme_large_cap_v20260613", "", False, "observation_only"),
                    _row("2024-01-02", "tw50_dynamic_constituents_v0", "", False, "observation_only"),
                    _row("2024-01-02", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool", top_asset_type="etf"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=1, asset_type="etf"),
                    _top("2024-01-02", "2882.TW", "formal_candidate", True, rank=2),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_independent_stock_expert(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            panel = pd.read_csv(output / "pool3_independent_stock_ranker_base_replay_panel.csv")
            pool3 = panel[panel["pool_id"] == "large_core_bluechip_v0"].iloc[0]
            self.assertEqual(pool3["top_ticker"], "2882.TW")
            self.assertTrue(_truthy(pool3["eligible_for_pool3_stock_vote"]))
            self.assertTrue(_truthy(pool3["eligible_for_exact_ticker_consensus"]))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["trade_decision_changed"])
            self.assertFalse(metadata["active_in_trade_decision"])

    def test_etf_never_becomes_pool3_stock_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool", top_asset_type="etf"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "00631L.TW", "market_exposure_tool", True, rank=1, asset_type="etf"),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_independent_stock_expert(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            summary = pd.read_csv(output / "pool3_independent_stock_expert_variant_summary.csv")
            self.assertEqual(set(summary["pool3_etf_stock_vote_rows"]), {0})
            self.assertEqual(set(summary["pool3_etf_exact_consensus_rows"]), {0})
            audit = pd.read_csv(output / "pool3_candidate_coverage_audit_daily.csv")
            self.assertEqual(audit.iloc[0]["coverage_state"], "candidate_empty")

    def test_coverage_audit_splits_gate_filter_from_low_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.csv"
            top = root / "top.csv"
            pd.DataFrame(
                [
                    _row("2024-01-02", "large_core_bluechip_v0", "2882.TW", False, "observation_only"),
                    _row("2024-01-03", "large_core_bluechip_v0", "2882.TW", False, "observation_only"),
                ]
            ).to_csv(replay, index=False)
            pd.DataFrame(
                [
                    _top("2024-01-02", "2882.TW", "observation_only", False, rank=1, score=0.4),
                    _top("2024-01-03", "2882.TW", "observation_only", False, rank=8, score=0.1),
                ]
            ).to_csv(top, index=False)

            output = run_pool3_independent_stock_expert(
                replay_panel_path=replay,
                top_candidates_path=top,
                output_dir=root / "out",
            )

            audit = pd.read_csv(output / "pool3_candidate_coverage_audit_daily.csv")
            states = dict(zip(audit["requested_signal_date"], audit["coverage_state"]))
            self.assertEqual(states["2024-01-02"], "filtered_out_by_gate")
            self.assertEqual(states["2024-01-03"], "no_edge_after_scoring")


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
        "gate_rule_id": "fixture",
        "gate_reason": "",
    }


def _top(
    date: str,
    ticker: str,
    selection_layer: str,
    eligible: bool,
    *,
    rank: int,
    score: float | None = None,
    asset_type: str = "stock",
) -> dict[str, object]:
    value = score if score is not None else max(0.0, 1.0 - rank / 10.0)
    return {
        "period": "2024",
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": "large_core_bluechip_v0",
        "ticker": ticker,
        "display": ticker,
        "rank": rank,
        "rank_score": value,
        "score": value,
        "asset_type": asset_type,
        "selection_layer": selection_layer,
        "eligible_for_pool_selection": eligible,
        "attack_gate_open": eligible,
        "base_pool_passed": eligible,
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    unittest.main()
