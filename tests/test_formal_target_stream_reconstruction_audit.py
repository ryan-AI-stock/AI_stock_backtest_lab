from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.formal_target_stream_reconstruction_audit import run_formal_target_stream_reconstruction_audit


class FormalTargetStreamReconstructionAuditTests(unittest.TestCase):
    def test_runner_outputs_dependency_matrix_without_generating_target_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            constituents = root / "tw50_constituents.csv"
            constituents.write_text(
                "effective_date,ticker,name,source,source_updated_at\n"
                "2025-06-23,2330.TW,台積電,current_proxy_snapshot,2026-06-29\n",
                encoding="utf-8",
            )
            price_root = root / "cache"
            price_root.mkdir()
            pd.DataFrame(
                {
                    "date": ["2014-11-03", "2023-12-29"],
                    "close": [100.0, 200.0],
                    "adj_close": [100.0, 200.0],
                }
            ).to_csv(price_root / "0050_TW.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2016-01-04", "2023-12-29"],
                    "close": [10.0, 20.0],
                    "adj_close": [10.0, 20.0],
                }
            ).to_csv(price_root / "00631L_TW.csv", index=False)
            supplement = root / "00631L_twse_stock_day_201411_201512.csv"
            pd.DataFrame(
                {
                    "date": ["2014-11-03", "2015-12-31"],
                    "ticker": ["00631L.TW", "00631L.TW"],
                    "open": [9.0, 11.0],
                    "high": [9.5, 11.5],
                    "low": [8.5, 10.5],
                    "close": [9.2, 11.2],
                    "volume": [1000, 2000],
                    "source": ["TWSE_STOCK_DAY", "TWSE_STOCK_DAY"],
                    "source_month": ["2014-11", "2015-12"],
                    "source_type": ["twse_stock_day_backfill", "twse_stock_day_backfill"],
                }
            ).to_csv(supplement, index=False)
            registry = root / "price_source_registry.csv"
            registry.write_text(
                "ticker,source_path,source_type,coverage_start,coverage_end,price_source_ready,strategy_ready,synthetic_used\n"
                f"00631L.TW,{supplement.as_posix()},twse_stock_day_backfill,2014-11-03,2015-12-31,true,false,false\n",
                encoding="utf-8",
            )
            output = root / "out"

            result = run_formal_target_stream_reconstruction_audit(
                constituents_path=constituents,
                price_roots=(price_root,),
                registry_path=registry,
                output_dir=output,
                repo_root=Path.cwd(),
            )

            self.assertEqual(result, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["target_stream_generated"])
            self.assertFalse(manifest["fake_target_stream_generated"])
            self.assertFalse(manifest["strategy_ready"])

            matrix = pd.read_csv(output / "formal_target_stream_dependency_matrix.csv")
            dependencies = set(matrix["dependency_name"])
            self.assertIn("tw50_pit_candidate_universe_by_date", dependencies)
            self.assertIn("pool1_candidate_ranking_scores", dependencies)
            self.assertIn("pool2_confirmation_state", dependencies)
            self.assertIn("formal_selector_target_contract", dependencies)
            self.assertIn("execution_price_ledger_inputs", dependencies)

            missing = pd.read_csv(output / "missing_inputs.csv")
            self.assertIn("pool1_candidate_ranking_scores", set(missing["dependency_name"]))
            self.assertIn("pool2_confirmation_state", set(missing["dependency_name"]))
            self.assertIn("previous_formal_target_contract", set(missing["dependency_name"]))

            reconstructable = pd.read_csv(output / "reconstructable_inputs.csv")
            self.assertIn("00631L_price_history", set(reconstructable["dependency_name"]))
            self.assertIn("formal_selector_target_contract", set(reconstructable["dependency_name"]))

            code_paths = pd.read_csv(output / "required_code_paths.csv")
            self.assertIn("formal_model_contract", set(code_paths["code_path_name"]))
            self.assertIn("stock_pool_observation_decision_first", set(code_paths["code_path_name"]))


if __name__ == "__main__":
    unittest.main()
