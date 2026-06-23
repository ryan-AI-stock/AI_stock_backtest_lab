from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_consensus_health_replay import (
    build_consensus_health_history_from_replay_panel,
    run_stock_pool_consensus_health_replay,
    summarize_consensus_health_periods,
)


class StockPoolConsensusHealthReplayTest(unittest.TestCase):
    def test_builds_report_only_health_history_from_replay_panel(self) -> None:
        replay = pd.DataFrame(
            [
                _row("2024", "2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                _row("2024", "2024-01-02", "tw50_dynamic_constituents_v0", "2330.TW", True, "formal_candidate"),
                _row("2024", "2024-01-02", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool"),
                _row("2024", "2024-01-03", "ai_theme_large_cap_v20260613", "2454.TW", True, "formal_candidate"),
                _row("2024", "2024-01-03", "tw50_dynamic_constituents_v0", "2330.TW", True, "formal_candidate"),
                _row("2024", "2024-01-03", "large_core_bluechip_v0", "2308.TW", True, "formal_candidate"),
                _row(
                    "2024",
                    "2024-01-04",
                    "ai_theme_large_cap_v20260613",
                    "2454.TW",
                    False,
                    "observation_only",
                    reason="個股攻擊閘門未開啟",
                ),
                _row("2024", "2024-01-04", "tw50_dynamic_constituents_v0", "00631L.TW", True, "market_exposure_tool"),
                _row("2024", "2024-01-04", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool"),
            ]
        )

        health, diagnostics = build_consensus_health_history_from_replay_panel(replay)

        self.assertEqual(len(health), 3)
        first = health[health["signal_date"] == "2024-01-02"].iloc[0]
        self.assertEqual(first["winner_ticker"], "2330.TW")
        self.assertEqual(first["decision_source"], "exact_2_of_3_ticker")
        second = health[health["signal_date"] == "2024-01-03"].iloc[0]
        self.assertEqual(second["result_state"], "divergent")
        self.assertEqual(second["decision_source"], "protocol_resolved_divergence")
        self.assertFalse(bool(second["decision_protocol_used"]))
        third = health[health["signal_date"] == "2024-01-04"].iloc[0]
        self.assertEqual(third["winner_ticker"], "00631L.TW")
        self.assertIn("observation_only_excluded", third["fake_consensus_flags"])
        blocked = diagnostics[diagnostics["selection_layer"] == "observation_only"].iloc[0]
        self.assertFalse(bool(blocked["eligible_vote"]))
        self.assertEqual(blocked["data_readiness_state"], "blocked")

    def test_period_summary_counts_health_buckets_and_protocol_candidates(self) -> None:
        health = pd.DataFrame(
            [
                _health("2024", "healthy", "exact_3_of_3_ticker", True, True, "consensus", 1.0, 0.0),
                _health("2024", "acceptable", "exact_2_of_3_ticker", True, True, "consensus", 1.0, 0.0),
                _health("2024", "unhealthy", "protocol_resolved_divergence", False, True, "divergent", 0.0, 0.0),
            ]
        )
        diagnostics = pd.DataFrame(
            [
                {"period": "2024", "pool_id": "tw50_dynamic_constituents_v0", "data_readiness_state": "blocked"},
                {"period": "2024", "pool_id": "large_core_bluechip_v0", "data_readiness_state": "ready"},
            ]
        )

        summary = summarize_consensus_health_periods(health, diagnostics)

        self.assertEqual(float(summary.loc[0, "exact_ticker_consensus_rate"]), 0.6667)
        self.assertEqual(float(summary.loc[0, "direction_consensus_rate"]), 1.0)
        self.assertEqual(float(summary.loc[0, "decision_protocol_candidate_rate"]), 0.3333)
        self.assertEqual(int(summary.loc[0, "pool2_blocked_or_partial_rows"]), 1)

    def test_skipped_rows_use_requested_signal_date_for_diagnostics(self) -> None:
        replay = pd.DataFrame(
            [
                _row("2022", "2022-01-03", "ai_theme_large_cap_v20260613", "00631L.TW", True, "market_exposure_tool"),
                _row("2022", "2022-01-03", "large_core_bluechip_v0", "2892.TW", True, "formal_candidate"),
                {
                    "period": "2022",
                    "requested_signal_date": "2022-01-03",
                    "signal_date": "",
                    "pool_id": "tw50_dynamic_constituents_v0",
                    "pool_name": "大型市場廣度池 v0",
                    "vote_group": "three_perspective_v1",
                    "status": "skipped",
                    "reason": "no_resolved_symbols",
                    "eligible_for_pool_selection": False,
                    "selection_layer": "no_selection",
                },
            ]
        )

        health, diagnostics = build_consensus_health_history_from_replay_panel(replay)

        self.assertEqual(len(health), 1)
        self.assertIn("tw50_dynamic_constituents_v0", set(diagnostics["pool_id"]))
        pool2 = diagnostics[diagnostics["pool_id"] == "tw50_dynamic_constituents_v0"].iloc[0]
        self.assertEqual(pool2["signal_date"], "2022-01-03")
        self.assertEqual(pool2["blocked_reason"], "no_resolved_symbols")

    def test_runner_can_use_existing_replay_panel_without_changing_trade_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_panel = root / "stock_pool_replay_panel.csv"
            pd.DataFrame(
                [
                    _row("2024", "2024-01-02", "ai_theme_large_cap_v20260613", "2330.TW", True, "formal_candidate"),
                    _row("2024", "2024-01-02", "tw50_dynamic_constituents_v0", "2330.TW", True, "formal_candidate"),
                    _row("2024", "2024-01-02", "large_core_bluechip_v0", "00631L.TW", True, "market_exposure_tool"),
                ]
            ).to_csv(replay_panel, index=False)

            output = run_stock_pool_consensus_health_replay(
                output_dir=root / "out",
                replay_panel_path=replay_panel,
                date_stride=5,
            )

            self.assertTrue((output / "stock_pool_consensus_health_history.csv").exists())
            self.assertTrue((output / "stock_pool_consensus_pool_diagnostics_history.csv").exists())
            manifest = (output / "manifest.json").read_text(encoding="utf-8")
            self.assertIn('"formal_model_changed": false', manifest)
            self.assertIn('"trade_decision_changed": false', manifest)


def _row(
    period: str,
    date: str,
    pool_id: str,
    ticker: str,
    eligible: bool,
    selection_layer: str,
    *,
    reason: str = "",
) -> dict:
    return {
        "period": period,
        "requested_signal_date": date,
        "signal_date": date,
        "pool_id": pool_id,
        "pool_name": pool_id,
        "vote_group": "three_perspective_v1",
        "status": "generated",
        "top_ticker": ticker,
        "top_display": ticker,
        "top_asset_type": "etf" if ticker in {"0050.TW", "00631L.TW"} else "stock",
        "rank_score": 1.0,
        "base_pool_passed": True,
        "attack_gate_open": selection_layer == "formal_candidate",
        "eligible_for_pool_selection": eligible,
        "selection_layer": selection_layer,
        "selection_reason": reason,
        "gate_rule_id": "test_gate",
        "gate_reason": reason,
        "action_state": "有合格模型目標",
        "decision_layer": "candidate_source",
        "active_in_trade_decision": False,
        "source_module": "stock_pool_observation",
    }


def _health(
    period: str,
    bucket: str,
    source: str,
    exact: bool,
    direction: bool,
    state: str,
    actionable_rate: float,
    protocol_used_rate: float,
) -> dict:
    return {
        "period": period,
        "consensus_health_bucket": bucket,
        "decision_source": source,
        "exact_ticker_consensus": exact,
        "direction_consensus": direction,
        "raw_consensus_state": state,
        "actionable_decision_rate": actionable_rate,
        "decision_protocol_used_rate": protocol_used_rate,
    }


if __name__ == "__main__":
    unittest.main()
