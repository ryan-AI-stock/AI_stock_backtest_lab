from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool3_radar_full_replay_acceptance import (
    run_pool3_radar_full_replay_acceptance,
)


class Pool3RadarFullReplayAcceptanceTest(unittest.TestCase):
    def test_rejects_partial_proxy_and_hard_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "full_replay"
            output = root / "out"
            source.mkdir()
            pd.DataFrame([{"date": "2024-01-02", "variant": "top10_base"}]).to_csv(
                source / "pool3_radar_weighted_basket_daily.csv",
                index=False,
            )
            pd.DataFrame([{"date": "2024-01-02", "data_status": "partial_proxy"}]).to_csv(
                source / "baseline_three_pool_daily_equity.csv",
                index=False,
            )
            pd.DataFrame([{"date": "2024-01-02", "changed": True}]).to_csv(
                source / "pool3_radar_full_replay_decision_diff.csv",
                index=False,
            )
            pd.DataFrame([{"ticker": "3260.TW", "share": 0.5}]).to_csv(
                source / "concentration_by_ticker_theme_month_quarter.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "period": "2024",
                        "variant": "ma200_radar20_00631l80_else_top10",
                        "total_return_pct": 47.8372,
                        "max_drawdown_pct": -33.1853,
                        "excess_vs_0050_pct": 2.7297,
                        "excess_vs_00631l_pct": -12.4496,
                    }
                ]
            ).to_csv(source / "pool3_radar_full_replay_summary.csv", index=False)
            (source / "readiness_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "partial_full_replay_pack",
                        "active_in_trade_decision": False,
                        "formal_model_modified": False,
                        "valuation_used": False,
                        "h3_used": False,
                        "can_core_absorb_as_formal_challenger": False,
                        "blockers": [
                            "baseline_three_pool_daily_equity is partial proxy expanded from stride20 vote panel, not formal daily replay",
                            "primary overlay daily basket is synthetic blend, transaction-cost realistic overlay trades not available",
                        ],
                        "rows": {"summary": 1},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (source / "formal_absorption_blocker_audit.md").write_text(
                "# Audit\n\nKeep report-only shadow.\n",
                encoding="utf-8",
            )

            result_dir = run_pool3_radar_full_replay_acceptance(
                full_replay_dir=source,
                output_dir=output,
            )

            manifest = json.loads((result_dir / "core_acceptance_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["core_decision"], "reject_formal_keep_report_only")
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["pool3_formal_vote_changed"])
            failed_ids = {row["check_id"] for row in manifest["failed_checks"]}
            self.assertIn("can_core_absorb_as_formal_challenger", failed_ids)
            self.assertIn("baseline_is_formal_daily_replay", failed_ids)
            self.assertIn("overlay_has_transaction_cost_accounting", failed_ids)
            self.assertIn("hard_gate_2024_mdd", failed_ids)
            self.assertIn("hard_gate_2024_excess_vs_0050", failed_ids)
            self.assertIn("hard_gate_2024_excess_vs_00631l", failed_ids)
            self.assertTrue((result_dir / "core_acceptance_checks.csv").exists())
            self.assertTrue((result_dir / "core_acceptance_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
