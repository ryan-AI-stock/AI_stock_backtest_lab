from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.rr_partial_switch_paper_trade_shadow import run_rr_partial_switch_paper_trade_shadow


class RRPartialSwitchPaperTradeShadowTest(unittest.TestCase):
    def test_runner_builds_shadow_tracker_without_formal_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_daily = root / "formal_daily.csv"
            price_cache = root / "prices"
            output = root / "out"
            price_cache.mkdir()
            input_rows = [
                _daily_row("2024-01-02", "", "cash", "hold", 0, 0, 1_000_000),
                _daily_row("2024-01-03", "00631L.TW", "00631L.TW", "buy", 1_000_000, 1425, 1_010_000),
                _daily_row("2024-01-04", "2454.TW", "2454.TW", "switch", 2_000_000, 5_000, 1_020_000),
                _daily_row("2024-01-05", "00631L.TW", "00631L.TW", "switch", 2_100_000, 5_200, 1_030_000),
                _daily_row("2024-01-08", "2454.TW", "2454.TW", "switch", 2_000_000, 5_000, 1_040_000),
                _daily_row("2024-01-09", "00631L.TW", "00631L.TW", "switch", 2_100_000, 5_200, 1_050_000),
            ]
            source = pd.DataFrame(input_rows)
            source.to_csv(formal_daily, index=False)
            before = formal_daily.read_text(encoding="utf-8")
            _write_price(price_cache, "00631L.TW", [10, 11, 12, 13, 14, 15, 16, 17])
            _write_price(price_cache, "2454.TW", [100, 102, 101, 105, 108, 110, 112, 114])
            _write_price(price_cache, "0050.TW", [100, 101, 102, 103, 104, 105, 106, 107])

            result = run_rr_partial_switch_paper_trade_shadow(
                formal_daily_path=formal_daily,
                price_cache_dir=price_cache,
                output_dir=output,
                initial_cash=1_000_000,
            )

            self.assertEqual(before, formal_daily.read_text(encoding="utf-8"))
            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["paper_trade_shadow_active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["formal_ready"])
            self.assertEqual(manifest["sample_gate_threshold"], 25)
            self.assertEqual(manifest["sample_gate_status"], "sample_limited_shadow_tracking")
            self.assertEqual(manifest["baseline_alignment"]["status"], "completed")

            contract = json.loads((result / "paper_trade_candidate_contract.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(contract["main_candidate"], "rr_partial_25_roundtrip_1_3")
            self.assertFalse(contract["sensitivity_allowed_to_replace_main"])

            event_log = pd.read_csv(result / "rr_event_shadow_log.csv")
            self.assertIn("event_id", event_log.columns)
            self.assertTrue(event_log["is_roundtrip_1_3"].astype(bool).any())
            self.assertFalse(event_log["forward_return_used_as_rule"].astype(bool).any())
            self.assertFalse(event_log["paper_trade_shadow_active_in_trade_decision"].astype(bool).any())

            sample = pd.read_csv(result / "sample_accumulation_status.csv")
            main = sample[sample["candidate_variant"].eq("rr_partial_25_roundtrip_1_3")].iloc[0]
            self.assertLess(int(main["total_event_count"]), 25)
            self.assertFalse(bool(main["ready_for_formal_readiness_recheck"]))

            gate = pd.read_csv(result / "event_count_gate_report.csv")
            self.assertFalse(gate["formal_ready"].astype(bool).any())

            forward_audit = pd.read_csv(result / "forward_return_rule_audit.csv").iloc[0]
            self.assertEqual(int(forward_audit["used_as_rule_count"]), 0)
            self.assertTrue(bool(forward_audit["pass"]))

            sensitivity = pd.read_csv(result / "sensitivity_any_1_3_report.csv")
            self.assertFalse(sensitivity["allowed_to_replace_main_candidate"].astype(bool).any())

            self.assertTrue((result / "paper_trade_shadow_summary_zh.md").exists())


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
