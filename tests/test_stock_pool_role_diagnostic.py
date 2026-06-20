from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.stock_pool_role_diagnostic import run_stock_pool_role_diagnostic


class StockPoolRoleDiagnosticTest(unittest.TestCase):
    def test_role_diagnostic_outputs_pool3_overlap_and_policy_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_dir = Path(temp_dir) / "replay"
            output_dir = Path(temp_dir) / "diagnostic"
            replay_dir.mkdir()
            _write_replay_panel(replay_dir / "stock_pool_replay_panel.csv")
            _write_forward_returns(replay_dir / "stock_pool_replay_forward_returns.csv")

            metadata = run_stock_pool_role_diagnostic(replay_dir=replay_dir, output_dir=output_dir)

            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((output_dir / "pool3_role_diagnostic.md").exists())
            summary = pd.read_csv(output_dir / "pool3_role_summary.csv")
            minority = pd.read_csv(output_dir / "pool3_minority_forward_summary.csv")
            policy = pd.read_csv(output_dir / "two_pool_vs_three_pool_policy_summary.csv")
            diagnostics = pd.read_csv(output_dir / "pool3_vote_diagnostics.csv")

            h20 = summary[summary["horizon"] == 20].iloc[0]
            self.assertAlmostEqual(h20["pool3_pool1_match_rate"], 0.0)
            self.assertAlmostEqual(h20["pool3_pool2_match_rate"], 1 / 3, places=6)
            self.assertAlmostEqual(h20["pool3_solo_rate"], 2 / 3, places=6)
            self.assertAlmostEqual(h20["pool3_breaks_tie_rate"], 1 / 3, places=6)
            self.assertAlmostEqual(h20["pool3_warns_two_pool_rate"], 1 / 3, places=6)

            self.assertEqual(int(minority[minority["horizon"] == 20].iloc[0]["sample_rows"]), 1)
            self.assertAlmostEqual(minority[minority["horizon"] == 20].iloc[0]["pool3_minus_two_pool_avg"], -0.03)
            self.assertGreaterEqual(int(policy[policy["horizon"] == 20].iloc[0]["three_pool_consensus_count"]), 2)
            self.assertIn("three_pool_minus_two_pool_paired_avg", policy.columns)
            self.assertIn("three_pool_minus_two_pool_unpaired_avg_delta", policy.columns)
            self.assertEqual(len(diagnostics), 9)


def _write_replay_panel(path: Path) -> None:
    rows = []
    # date1: pool1 and pool2 agree on AAA; pool3 warns with CCC.
    rows.extend(_panel_rows("2026-01-02", "AAA", "AAA", "CCC"))
    # date2: pool1 AAA, pool2 BBB, pool3 BBB breaks the tie.
    rows.extend(_panel_rows("2026-01-05", "AAA", "BBB", "BBB"))
    # date3: pool3 is solo and no consensus.
    rows.extend(_panel_rows("2026-01-06", "AAA", "BBB", "CCC"))
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _panel_rows(signal_date: str, pool1: str, pool2: str, pool3: str) -> list[dict[str, object]]:
    return [
        _panel_row(signal_date, "ai_theme_large_cap_v20260613", pool1),
        _panel_row(signal_date, "tw50_dynamic_constituents_v0", pool2),
        _panel_row(signal_date, "large_core_bluechip_v0", pool3),
    ]


def _panel_row(signal_date: str, pool_id: str, ticker: str) -> dict[str, object]:
    return {
        "period": "test",
        "requested_signal_date": signal_date,
        "signal_date": signal_date,
        "status": "generated",
        "pool_id": pool_id,
        "pool_name": pool_id,
        "top_ticker": ticker,
        "top_display": ticker,
        "selection_layer": "formal_candidate",
        "eligible_for_pool_selection": True,
    }


def _write_forward_returns(path: Path) -> None:
    returns = {
        ("2026-01-02", "ai_theme_large_cap_v20260613", "AAA"): 0.10,
        ("2026-01-02", "tw50_dynamic_constituents_v0", "AAA"): 0.10,
        ("2026-01-02", "large_core_bluechip_v0", "CCC"): 0.07,
        ("2026-01-05", "ai_theme_large_cap_v20260613", "AAA"): 0.02,
        ("2026-01-05", "tw50_dynamic_constituents_v0", "BBB"): 0.08,
        ("2026-01-05", "large_core_bluechip_v0", "BBB"): 0.08,
        ("2026-01-06", "ai_theme_large_cap_v20260613", "AAA"): 0.01,
        ("2026-01-06", "tw50_dynamic_constituents_v0", "BBB"): 0.03,
        ("2026-01-06", "large_core_bluechip_v0", "CCC"): 0.04,
    }
    rows = []
    for (signal_date, pool_id, ticker), value in returns.items():
        for horizon in (20, 60, 120):
            rows.append(
                {
                    "period": "test",
                    "requested_signal_date": signal_date,
                    "signal_date": signal_date,
                    "pool_id": pool_id,
                    "ticker": ticker,
                    "horizon": horizon,
                    "forward_status": "ready",
                    "forward_return": value,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    unittest.main()
