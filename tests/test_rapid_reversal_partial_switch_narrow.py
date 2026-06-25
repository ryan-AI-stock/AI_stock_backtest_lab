from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.partial_execution_ledger import _normalize_formal_daily
from backtest_lab.rapid_reversal_partial_switch_narrow import (
    build_rapid_reversal_event_labels,
    run_rapid_reversal_partial_switch_narrow,
)


class RapidReversalPartialSwitchNarrowTest(unittest.TestCase):
    def test_labels_cover_any_and_roundtrip_windows(self) -> None:
        frame = _normalize_formal_daily(
            pd.DataFrame(
                [
                    _daily_row("2024-01-02", "A.TW", "cash", "buy", 0, 0, 1_000_000),
                    _daily_row("2024-01-03", "B.TW", "B.TW", "switch", 0, 0, 1_000_000),
                    _daily_row("2024-01-04", "A.TW", "A.TW", "switch", 0, 0, 1_000_000),
                    _daily_row("2024-01-05", "C.TW", "C.TW", "switch", 0, 0, 1_000_000),
                    _daily_row("2024-01-08", "D.TW", "D.TW", "switch", 0, 0, 1_000_000),
                ]
            )
        )
        labels = build_rapid_reversal_event_labels(frame)
        first = labels[labels["date"].eq("2024-01-02")].iloc[0]
        second = labels[labels["date"].eq("2024-01-03")].iloc[0]
        self.assertTrue(bool(first["rapid_reversal_any_1_3"]))
        self.assertTrue(bool(first["rapid_reversal_any_1_2"]))
        self.assertFalse(bool(first["rapid_reversal_roundtrip_1_3"]))
        self.assertTrue(bool(second["rapid_reversal_roundtrip_1_3"]))
        self.assertTrue(bool(second["rapid_reversal_roundtrip_1_2"]))

    def test_runner_outputs_diagnostic_narrow_ledgers(self) -> None:
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
                    _daily_row("2024-01-08", "2454.TW", "2454.TW", "switch", 2_000_000, 5_000, 1_040_000),
                    _daily_row("2024-01-09", "00631L.TW", "00631L.TW", "switch", 2_100_000, 5_200, 1_050_000),
                ]
            ).to_csv(formal_daily, index=False)
            _write_price(price_cache, "00631L.TW", [10, 11, 12, 13, 14, 15, 16, 17])
            _write_price(price_cache, "2454.TW", [100, 102, 101, 105, 108, 110, 112, 114])
            _write_price(price_cache, "0050.TW", [100, 101, 102, 103, 104, 105, 106, 107])

            result = run_rapid_reversal_partial_switch_narrow(
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
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["valuation_used"])
            self.assertFalse(manifest["h3_used"])
            self.assertFalse(manifest["pool3_shadow_used"])
            self.assertFalse(manifest["final_decision_label_used"])

            variants = pd.read_csv(result / "variant_parameter_matrix.csv")
            self.assertIn("rr_partial_50_roundtrip_1_2", variants["variant_id"].tolist())

            labels = pd.read_csv(result / "rapid_reversal_event_labels.csv")
            self.assertIn("rapid_reversal_any_1_3", labels.columns)
            self.assertTrue(labels["rapid_reversal_any_1_3"].astype(bool).any())

            forward = pd.read_csv(result / "forward_return_evaluation_labels.csv")
            self.assertIn("forward_return_used_as_rule", forward.columns)
            self.assertFalse(forward["forward_return_used_as_rule"].astype(bool).any())

            daily = pd.read_csv(result / "narrow_partial_execution_daily_ledger.csv")
            self.assertLessEqual(float(pd.to_numeric(daily["weight_sum"], errors="coerce").max()), 1.0001)
            self.assertGreaterEqual(float(pd.to_numeric(daily["cash_value"], errors="coerce").min()), 0.0)
            self.assertFalse(daily["execution_diagnostic_active_in_trade_decision"].astype(bool).any())

            trades = pd.read_csv(result / "narrow_partial_execution_trade_ledger.csv")
            self.assertGreater(float(pd.to_numeric(trades["transaction_cost"], errors="coerce").sum()), 0.0)

            self.assertTrue((result / "baseline_vs_narrow_challenger_summary_zh.md").exists())


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
