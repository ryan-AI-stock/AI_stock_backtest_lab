from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_primary_risk_overlay_challenger import (
    VARIANT_POOL1_PRIMARY,
    VARIANT_POOL2_VETO,
    VARIANT_POOL2_WARNING,
    VARIANT_POOL23_REPORT,
    run_pool1_primary_risk_overlay_challenger,
)


class Pool1PrimaryRiskOverlayChallengerTest(unittest.TestCase):
    def test_warning_overlays_do_not_change_equity_and_veto_records_reason(self) -> None:
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

            result = run_pool1_primary_risk_overlay_challenger(
                formal_replay_dir=formal,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])
            self.assertFalse(manifest["report_only_labels_used_in_performance"])
            self.assertTrue(manifest["same_date_range_for_variants"])

            daily = pd.read_csv(result / "daily_equity_by_variant.csv")
            primary_equity = daily[daily["variant"].eq(VARIANT_POOL1_PRIMARY)]["equity"].tolist()
            warning_equity = daily[daily["variant"].eq(VARIANT_POOL2_WARNING)]["equity"].tolist()
            report_equity = daily[daily["variant"].eq(VARIANT_POOL23_REPORT)]["equity"].tolist()
            self.assertEqual(warning_equity, primary_equity)
            self.assertEqual(report_equity, primary_equity)

            veto = pd.read_csv(result / "veto_event_panel.csv")
            self.assertIn(VARIANT_POOL2_VETO, set(veto["variant"]))
            self.assertIn("pool2_disagrees_with_pool1", ";".join(veto["risk_veto_reason"].astype(str).tolist()))
            self.assertIn("vetoed_target", veto.columns)

            risk = pd.read_csv(result / "risk_overlay_event_panel.csv")
            pool23 = risk[risk["variant"].eq("pool1_primary_pool2_pool3_risk_veto")]
            self.assertFalse(pool23["pool3_shadow_used_as_formal"].any() if "pool3_shadow_used_as_formal" in pool23.columns else False)

            entry = pd.read_csv(result / "entry_without_exit_confirmation_panel.csv")
            self.assertIn("entry_signal_without_exit_confirmation", entry.columns)
            target_drop = pd.read_csv(result / "target_drop_from_top3_diagnostics.csv")
            self.assertIn("target_drop_from_top3_next_3d", target_drop.columns)


def _write_decision(root: Path) -> None:
    pd.DataFrame(
        [
            _decision_row("2024-01-02", "AAA.TW", "BBB.TW", "BBB.TW", ""),
            _decision_row("2024-01-03", "AAA.TW", "AAA.TW", "BBB.TW", "AAA.TW"),
            _decision_row("2024-01-04", "BBB.TW", "AAA.TW", "AAA.TW", ""),
            _decision_row("2024-01-05", "BBB.TW", "BBB.TW", "AAA.TW", "BBB.TW"),
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
