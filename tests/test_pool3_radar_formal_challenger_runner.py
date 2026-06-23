from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_radar_formal_challenger_runner import (
    PRIMARY_CANDIDATE,
    run_pool3_radar_formal_challenger_runner,
)


class Pool3RadarFormalChallengerRunnerTest(unittest.TestCase):
    def test_runner_builds_partial_contract_without_promoting_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opportunity = root / "opportunity"
            shadow = root / "shadow"
            attack = root / "attack"
            output = root / "out"
            opportunity.mkdir()
            shadow.mkdir()
            attack.mkdir()
            pd.DataFrame(
                [
                    {
                        "timestamp": "2026-06-23 09:54:14",
                        "period": "2024",
                        "variant_id": PRIMARY_CANDIDATE,
                        "total_return_pct": 47.84,
                        "max_drawdown_pct": -33.19,
                        "benchmark_0050_return_pct": 45.85,
                        "benchmark_00631l_return_pct": 57.10,
                        "excess_vs_0050_pct": 1.99,
                        "excess_vs_00631l_pct": -9.26,
                    },
                    {
                        "timestamp": "2026-06-23 09:54:14",
                        "period": "2025",
                        "variant_id": PRIMARY_CANDIDATE,
                        "total_return_pct": 56.07,
                        "max_drawdown_pct": -18.0,
                        "benchmark_0050_return_pct": 36.31,
                        "benchmark_00631l_return_pct": 60.0,
                        "excess_vs_0050_pct": 19.76,
                        "excess_vs_00631l_pct": -3.93,
                    },
                ]
            ).to_csv(opportunity / "overlay_performance.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "period": "2024",
                        "signal_date": "2024-01-02",
                        "scenario": "baseline_existing_pool3",
                        "pool1_vote": "00631L.TW",
                        "pool2_vote": "2327.TW",
                        "pool3_vote": "00631L.TW",
                        "pool3_shadow_risk_on_0050_ma200": True,
                        "radar_top1_ticker": "",
                        "radar_top1_theme": "",
                        "radar_top1_weight": "",
                        "result_state": "consensus",
                        "winner_ticker": "00631L.TW",
                        "winner_vote_count": 2,
                    },
                    {
                        "period": "2024",
                        "signal_date": "2024-01-02",
                        "scenario": "pool3_radar_top1_always",
                        "pool1_vote": "00631L.TW",
                        "pool2_vote": "2327.TW",
                        "pool3_vote": "3260.TW",
                        "pool3_shadow_risk_on_0050_ma200": True,
                        "radar_top1_ticker": "3260.TW",
                        "radar_top1_theme": "記憶體",
                        "radar_top1_weight": 0.2,
                        "result_state": "no_vote",
                        "winner_ticker": "",
                        "winner_vote_count": 0,
                    },
                ]
            ).to_csv(shadow / "three_pool_shadow_vote_panel.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "period": "2024",
                        "variant_id": "top3_theme_capital_flow_top10_weekly",
                        "total_return_pct": -3.6,
                        "max_drawdown_pct": -6.5,
                        "trade_count": 10,
                        "benchmark_0050_return_pct": 44.8,
                        "benchmark_00631l_return_pct": 57.8,
                        "excess_vs_0050_pct": -48.4,
                        "excess_vs_00631l_pct": -61.4,
                    }
                ]
            ).to_csv(attack / "yearly_performance.csv", index=False)
            pd.DataFrame([{"ticker": "3260.TW", "share": 0.5}]).to_csv(attack / "ticker_concentration.csv", index=False)
            pd.DataFrame([{"theme": "記憶體", "share": 0.8}]).to_csv(attack / "theme_concentration.csv", index=False)
            (attack / "readiness_summary.json").write_text(
                json.dumps({"status": "partial", "blocking_issues": ["formal_top3_partial"]}, ensure_ascii=False),
                encoding="utf-8",
            )

            result_dir = run_pool3_radar_formal_challenger_runner(
                opportunity_overlay_dir=opportunity,
                three_pool_shadow_dir=shadow,
                attack_pool_dir=attack,
                output_dir=output,
            )

            metadata = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "partial_contract")
            self.assertFalse(metadata["active_in_trade_decision"])
            self.assertFalse(metadata["formal_model_changed"])
            self.assertFalse(metadata["valuation_used"])
            self.assertFalse(metadata["h3_used"])
            self.assertFalse(metadata["full_replay_ready"])
            self.assertIn("daily weighted basket holdings", metadata["full_replay_blockers"][0])
            self.assertEqual(metadata["hard_gate_2024"][0]["status"], "needs_research_review")

            diff = pd.read_csv(result_dir / "decision_diff_panel.csv")
            radar_row = diff[diff["scenario"] == "pool3_radar_top1_always"].iloc[0]
            self.assertTrue(bool(radar_row["changed_pool3_vote_from_baseline"]))
            self.assertFalse(bool(radar_row["active_in_trade_decision"]))

            self.assertTrue((result_dir / "baseline_vs_challengers.csv").exists())
            self.assertTrue((result_dir / "concentration_summary.csv").exists())
            self.assertTrue((result_dir / "hard_gate_2024.csv").exists())
            self.assertTrue((result_dir / "final_summary_zh.md").exists())


if __name__ == "__main__":
    unittest.main()
