from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_pool2_veto_robustness import run_pool1_pool2_veto_robustness


class Pool1Pool2VetoRobustnessTest(unittest.TestCase):
    def test_outputs_robustness_contract_without_formal_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            challenger = root / "challenger"
            prices = root / "prices"
            output = root / "out"
            challenger.mkdir()
            prices.mkdir()
            _write_challenger_files(challenger)
            _write_prices(prices, "00631L.TW", [10, 11, 12, 13, 14, 15, 16, 17])
            _write_prices(prices, "0050.TW", [20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5])
            _write_prices(prices, "AAA.TW", [30, 31, 29, 32, 33, 34, 35, 36])

            result = run_pool1_pool2_veto_robustness(
                challenger_dir=challenger,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])

            exposure = pd.read_csv(result / "00631L_exposure_breakdown.csv")
            self.assertIn("00631L_position_day_share", exposure.columns)
            self.assertIn("00631L_trade_count", exposure.columns)

            veto = pd.read_csv(result / "vetoed_event_forward_outcome.csv")
            self.assertIn("date", veto.columns)
            self.assertIn("veto_reason", veto.columns)
            self.assertIn("vetoed_target", veto.columns)
            self.assertIn("actual_target_or_holding", veto.columns)
            self.assertIn("vetoed_excess_vs_0050_20d", veto.columns)
            self.assertFalse(veto["uses_forward_return_as_rule"].map(bool).any())

            exclusion = pd.read_csv(result / "contribution_exclusion_tests.csv")
            self.assertTrue(exclusion["period_label"].astype(str).str.contains("exclude_00631L").any())


def _write_challenger_files(root: Path) -> None:
    daily_rows = []
    target_rows = []
    for variant in ["pool1_primary_pool2_risk_veto", "pool1_primary_no_overlay", "current_formal_three_pool_baseline"]:
        for index in range(6):
            date = (pd.Timestamp("2024-01-02") + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
            ticker = "00631L.TW" if index < 4 else "AAA.TW"
            daily_rows.append(
                {
                    "variant": variant,
                    "date": date,
                    "period": "2024_now",
                    "position_ticker": ticker,
                    "winner_ticker": ticker,
                    "equity": 1_000_000 + index * 10_000,
                    "drawdown": 0,
                    "transaction_cost": 100 if index in {0, 4} else 0,
                    "turnover": 10_000 if index in {0, 4} else 0,
                    "action": "buy" if index in {0, 4} else "hold",
                }
            )
            target_rows.append(
                {
                    "variant": variant,
                    "date": date,
                    "period": "2024_now",
                    "formal_target": ticker,
                    "position_ticker": ticker,
                    "pool1_vote": ticker,
                    "pool2_vote": "AAA.TW",
                    "pool3_vote": "",
                    "entry_signal_without_exit_confirmation": index == 2,
                }
            )
    pd.DataFrame(daily_rows).to_csv(root / "daily_equity_by_variant.csv", index=False)
    pd.DataFrame(target_rows).to_csv(root / "daily_target_by_variant.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "pool1_primary_pool2_risk_veto",
                "period": "2024_now",
                "date": "2024-01-02",
                "vetoed_target": "00631L.TW",
                "pool2_vote": "AAA.TW",
                "pool3_vote": "",
                "variant_target": "",
                "risk_veto_reason": "pool2_disagrees_with_pool1",
            }
        ]
    ).to_csv(root / "veto_event_panel.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "pool1_primary_pool2_risk_veto",
                "date": "2024-01-02",
                "ticker": "00631L.TW",
                "action": "buy",
                "costs": 100,
            }
        ]
    ).to_csv(root / "trade_ledger_by_variant.csv", index=False)


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
