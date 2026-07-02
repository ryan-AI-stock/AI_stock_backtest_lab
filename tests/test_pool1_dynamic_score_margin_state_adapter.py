import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_dynamic_score_margin_state_adapter import run_pool1_dynamic_score_margin_state_adapter


class Pool1DynamicScoreMarginStateAdapterTest(unittest.TestCase):
    def _price_frame(self, start: str = "2014-10-01", periods: int = 90) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = [100 + index for index in range(periods)]
        return pd.DataFrame(
            {
                "date": [date.strftime("%Y-%m-%d") for date in dates],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": [1000] * periods,
                "dividend": [0.0] * periods,
                "stock_split": [0.0] * periods,
            }
        )

    def test_builds_score_margin_panel_but_blocks_formal_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel"
            dynamic = root / "dynamic"
            cache = root / "cache"
            output = root / "out"
            panel.mkdir()
            dynamic.mkdir()
            cache.mkdir()

            signal_date = "2015-01-28"
            (panel / "manifest.json").write_text(
                json.dumps({"date_start": signal_date, "date_end": signal_date}, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "date": signal_date,
                        "candidate_ticker": "2330.TW",
                        "candidate_name": "台積電",
                        "score": 1.0,
                        "rank": 1,
                    }
                ]
            ).to_csv(panel / "pool1_daily_candidate_ranking_panel.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": signal_date,
                        "available_universe_count": 1,
                        "candidate_tickers": "2330.TW",
                    }
                ]
            ).to_csv(dynamic / "dynamic_universe_state_replay_coverage.csv", index=False)
            self._price_frame().to_csv(cache / "0050_TW.csv", index=False)

            result = run_pool1_dynamic_score_margin_state_adapter(
                panel_dir=panel,
                dynamic_universe_dir=dynamic,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
            )
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["dynamic_score_margin_panel_ready"])
            self.assertFalse(manifest["dynamic_attack_gate_formal_ready"])
            self.assertEqual(manifest["equivalence_regression_2022plus_status"], "blocked_missing_2022_dynamic_equivalence_runner")
            self.assertFalse(manifest["no_target_cash_all_applied"])

            score = pd.read_csv(output / "dynamic_score_margin_panel.csv").iloc[0]
            self.assertTrue(bool(score["fallback_0050_score_ready"]))
            self.assertNotEqual(str(score["fallback_0050_score"]), "")
            self.assertNotEqual(str(score["score_margin"]), "")
            self.assertFalse(bool(score["source_formal_ready"]))

            equivalence = pd.read_csv(output / "equivalence_regression_2022plus.csv").iloc[0]
            self.assertEqual(equivalence["status"], "blocked_not_run")

            blocked = pd.read_csv(output / "blocked_signal_rows.csv").iloc[0]
            self.assertFalse(bool(blocked["source_formal_ready"]))
            self.assertFalse(bool(blocked["no_target_cash_all_applied"]))


if __name__ == "__main__":
    unittest.main()
