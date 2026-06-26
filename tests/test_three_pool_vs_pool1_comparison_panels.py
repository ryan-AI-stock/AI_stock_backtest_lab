from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.three_pool_vs_pool1_comparison_panels import run_three_pool_vs_pool1_comparison_panels


class ThreePoolVsPool1ComparisonPanelsTest(unittest.TestCase):
    def test_builds_same_date_range_and_keeps_overlays_out_of_performance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = root / "formal"
            prices = root / "prices"
            output = root / "out"
            formal.mkdir()
            prices.mkdir()
            _write_decision(formal)
            _write_baseline_daily(formal)
            _write_prices(prices, "AAA.TW", [10, 11, 12, 13, 14, 15])
            _write_prices(prices, "BBB.TW", [20, 19, 18, 17, 16, 15])
            report_boundary = root / "report_boundary.csv"
            pd.DataFrame(
                [
                    {
                        "final_decision_user_reading_state": "strong_consensus_supported",
                        "final_decision_label_active_in_trade_decision": False,
                    }
                ]
            ).to_csv(report_boundary, index=False)
            rr_shadow = root / "rr_shadow.csv"
            pd.DataFrame([{"candidate": "rr_partial_25_roundtrip_1_3", "formal_ready": False}]).to_csv(rr_shadow, index=False)

            result = run_three_pool_vs_pool1_comparison_panels(
                formal_replay_dir=formal,
                pool_diagnostics_path=root / "missing_pool_signal.csv",
                report_boundary_path=report_boundary,
                rr_shadow_path=rr_shadow,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])
            self.assertFalse(manifest["report_only_labels_used_in_performance"])
            self.assertFalse(manifest["rr_partial_switch_used_in_performance"])
            self.assertTrue(manifest["same_date_range_for_variants"])
            self.assertTrue(manifest["same_cost_model_for_variants"])
            self.assertEqual(manifest["latest_complete_common_date"], "2024-01-05")

            daily = pd.read_csv(result / "daily_equity_by_variant.csv")
            baseline = daily[daily["variant"].eq("current_formal_three_pool_baseline")]["equity"].tolist()
            labels = daily[daily["variant"].eq("three_pool_with_report_only_labels")]["equity"].tolist()
            execution = daily[daily["variant"].eq("three_pool_with_execution_shadow_diagnostics")]["equity"].tolist()
            self.assertEqual(labels, baseline)
            self.assertEqual(execution, baseline)

            targets = pd.read_csv(result / "daily_target_by_variant.csv")
            pool1 = targets[targets["variant"].eq("pool1_only_formal_replay")]
            self.assertEqual(pool1.iloc[0]["formal_target"], "AAA.TW")
            self.assertTrue(bool(pool1.iloc[0]["entry_signal_without_exit_confirmation"]))
            self.assertIn("target_drop_from_top3_next_1d", pd.read_csv(result / "target_drop_from_top3_diagnostics.csv").columns)

            period = pd.read_csv(result / "period_performance_by_variant.csv")
            self.assertIn("pool1_only_formal_replay", set(period["variant"]))
            self.assertIn("current_formal_three_pool_baseline", set(period["variant"]))


def _write_decision(root: Path) -> None:
    pd.DataFrame(
        [
            _decision_row("2024-01-02", "AAA.TW", "BBB.TW", "", ""),
            _decision_row("2024-01-03", "AAA.TW", "AAA.TW", "", "AAA.TW"),
            _decision_row("2024-01-04", "BBB.TW", "AAA.TW", "", ""),
            _decision_row("2024-01-05", "BBB.TW", "BBB.TW", "", "BBB.TW"),
        ]
    ).to_csv(root / "formal_three_pool_decision_panel.csv", index=False)


def _decision_row(date: str, p1: str, p2: str, p3: str, winner: str) -> dict[str, object]:
    return {
        "period": "2024_now",
        "date": date,
        "pool1_vote": p1,
        "pool2_vote": p2,
        "pool3_vote": p3,
        "consensus_state": "consensus" if winner else "divergent",
        "winner_ticker": winner,
        "eligible_vote_count": 2,
    }


def _write_baseline_daily(root: Path) -> None:
    pd.DataFrame(
        [
            _daily_row("2024-01-02", "", "cash", 1_000_000, "hold"),
            _daily_row("2024-01-03", "AAA.TW", "AAA.TW", 1_010_000, "buy", 1000, 10),
            _daily_row("2024-01-04", "", "AAA.TW", 1_020_000, "hold"),
            _daily_row("2024-01-05", "BBB.TW", "BBB.TW", 1_030_000, "switch", 2000, 20),
        ]
    ).to_csv(root / "baseline_three_pool_formal_daily_equity.csv", index=False)


def _daily_row(date: str, winner: str, position: str, equity: float, action: str, turnover: float = 0, cost: float = 0) -> dict[str, object]:
    return {
        "date": date,
        "period": "2024_now",
        "pool1_vote": "",
        "pool2_vote": "",
        "pool3_vote": "",
        "consensus_state": "",
        "winner_ticker": winner,
        "position_ticker": position,
        "cash": 0,
        "equity": equity,
        "drawdown": 0,
        "turnover": turnover,
        "transaction_cost": cost,
        "action": action,
        "data_status": "formal_daily_replay",
    }


def _write_prices(root: Path, ticker: str, closes: list[float]) -> None:
    rows = []
    for offset, close in enumerate(closes):
        rows.append(
            {
                "date": (pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": close,
                "close": close,
                "adj_close": close,
            }
        )
    pd.DataFrame(rows).to_csv(root / f"{ticker.replace('.', '_')}.csv", index=False)


if __name__ == "__main__":
    unittest.main()
