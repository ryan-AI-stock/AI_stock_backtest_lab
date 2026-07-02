import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS, TW50_BENCHMARK
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import run_pool1_dynamic_adapter_2022_equivalence


class Pool1DynamicAdapter2022EquivalenceTest(unittest.TestCase):
    def _price_frame(self, start: str = "2021-01-01", periods: int = 390, offset: float = 0.0) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = [100.0 + offset + index * (1.0 + offset / 100.0) for index in range(periods)]
        return pd.DataFrame(
            {
                "date": [date.strftime("%Y-%m-%d") for date in dates],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": [1000000] * periods,
                "dividend": [0.0] * periods,
                "stock_split": [0.0] * periods,
            }
        )

    def test_builds_equivalence_package_without_formal_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            formal = root / "formal"
            output = root / "out"
            cache.mkdir()
            formal.mkdir()

            tickers = sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK})
            for index, ticker in enumerate(tickers):
                self._price_frame(offset=float(index)).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "signal_date": "2022-04-01",
                        "execution_date": "2022-04-04",
                        "formal_target": "00631L.TW",
                        "target_weights": '{"00631L.TW": 1.0}',
                        "risk_off_state": "formal_target_active",
                        "pool1_top_candidate": "00631L.TW",
                    },
                    {
                        "signal_date": "2022-04-04",
                        "execution_date": "2022-04-05",
                        "formal_target": "00631L.TW",
                        "target_weights": '{"00631L.TW": 1.0}',
                        "risk_off_state": "formal_target_active",
                        "pool1_top_candidate": "00631L.TW",
                    },
                ]
            ).to_csv(formal / "formal_long_range_target_stream.csv", index=False)

            result = run_pool1_dynamic_adapter_2022_equivalence(
                formal_long_range_dir=formal,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
                start_date="2022-04-01",
                end_date="2022-04-04",
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["raw_diagnostic_pass_used_as_formal_target"])
            self.assertIn("equivalence_pass", manifest)

            self.assertTrue((output / "dynamic_adapter_2022_score_margin_panel.csv").exists())
            self.assertTrue((output / "formal_pool1_reference_stream.csv").exists())
            self.assertTrue((output / "equivalence_regression_2022plus.csv").exists())
            self.assertTrue((output / "equivalence_summary.csv").exists())

            regression = pd.read_csv(output / "equivalence_regression_2022plus.csv")
            self.assertIn("row_match_for_formal_readiness", regression.columns)
            self.assertIn("raw_gate_matches_reference_attack_gate", regression.columns)

            source_decision = pd.read_csv(output / "proxy_or_formal_source_decision.csv")
            self.assertIn("dynamic_adapter_2014_2021_promotion", set(source_decision["source_layer"]))


if __name__ == "__main__":
    unittest.main()
