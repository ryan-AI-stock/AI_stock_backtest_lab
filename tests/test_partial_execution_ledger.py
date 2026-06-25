from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.partial_execution_ledger import run_partial_execution_ledger


class PartialExecutionLedgerTest(unittest.TestCase):
    def test_runner_builds_diagnostic_ledgers_without_formal_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_daily = root / "formal_daily.csv"
            price_cache = root / "prices"
            output = root / "out"
            price_cache.mkdir()
            pd.DataFrame(
                [
                    _daily_row("2024-01-02", "", "cash", "hold", 0, 0, 1_000_000),
                    _daily_row("2024-01-03", "00631L.TW", "00631L.TW", "buy", 1_000_000, 1425, 1_010_000),
                    _daily_row("2024-01-04", "2454.TW", "2454.TW", "switch", 2_000_000, 5_000, 1_020_000),
                    _daily_row("2024-01-05", "00631L.TW", "00631L.TW", "switch", 2_100_000, 5_200, 1_030_000),
                    _daily_row("2024-01-08", "00631L.TW", "00631L.TW", "hold", 0, 0, 1_040_000),
                ]
            ).to_csv(formal_daily, index=False)
            _write_price(price_cache, "00631L.TW", [10, 11, 12, 13, 14, 15, 16])
            _write_price(price_cache, "2454.TW", [100, 102, 101, 105, 108, 110, 112])
            _write_price(price_cache, "0050.TW", [100, 101, 102, 103, 104, 105, 106])

            result = run_partial_execution_ledger(
                formal_daily_path=formal_daily,
                price_cache_dir=price_cache,
                output_dir=output,
                initial_cash=1_000_000,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["execution_diagnostic_active_in_trade_decision"])
            self.assertFalse(manifest["proxy_performance"])
            self.assertEqual(manifest["baseline_alignment"]["status"], "completed")

            daily = pd.read_csv(result / "partial_execution_daily_ledger.csv")
            self.assertIn("baseline_full_rotation", daily["variant_id"].tolist())
            self.assertLessEqual(float(pd.to_numeric(daily["weight_sum"], errors="coerce").max()), 1.0001)
            self.assertGreaterEqual(float(pd.to_numeric(daily["cash_value"], errors="coerce").min()), 0.0)
            self.assertFalse(daily["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            trades = pd.read_csv(result / "partial_execution_trade_ledger.csv")
            self.assertGreater(float(pd.to_numeric(trades["transaction_cost"], errors="coerce").sum()), 0.0)
            self.assertFalse(trades["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            blocked = pd.read_csv(result / "blocked_variants.csv")
            self.assertIn("sell_first_then_buy_global", blocked["variant_id"].tolist())
            self.assertIn("pause_on_conflict", blocked["variant_id"].tolist())
            self.assertTrue(blocked["blocked_reason"].astype(str).str.len().gt(0).all())

            period = pd.read_csv(result / "partial_execution_period_performance.csv")
            self.assertIn("full_2022_2026", period["period"].tolist())
            self.assertIn("benchmark_00631l_return_pct", period.columns)

            self.assertTrue((result / "baseline_vs_partial_execution_summary_zh.md").exists())


def _daily_row(date: str, winner: str, position: str, action: str, turnover: float, cost: float, equity: float) -> dict:
    return {
        "date": date,
        "period": "2024_now",
        "pool1_vote": winner,
        "pool2_vote": winner,
        "pool3_vote": "",
        "consensus_state": "consensus" if winner else "divergent",
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


def _write_price(cache: Path, ticker: str, closes: list[float]) -> None:
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": (pd.Timestamp("2024-01-02") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                "open": close,
                "close": close,
                "adj_close": close,
            }
        )
    pd.DataFrame(rows).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)


if __name__ == "__main__":
    unittest.main()
