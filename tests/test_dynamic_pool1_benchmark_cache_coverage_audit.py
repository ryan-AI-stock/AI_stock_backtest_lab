import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_benchmark_cache_coverage_audit import run_benchmark_cache_coverage_audit


class DynamicPool1BenchmarkCacheCoverageAuditTest(unittest.TestCase):
    def test_audits_missing_months_and_join_repair_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = root / "context"
            context.mkdir()
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2020-01",
                        "rows": 10,
                        "explicit_0050_ready_rate": 0.0,
                        "explicit_00631l_ready_rate": 0.0,
                    },
                    {
                        "candidate_month": "2024-01",
                        "rows": 10,
                        "explicit_0050_ready_rate": 1.0,
                        "explicit_00631l_ready_rate": 1.0,
                    },
                ]
            ).to_csv(context / "benchmark_readiness_summary.csv", index=False)
            primary = root / "backtest_cache"
            primary.mkdir()
            pd.DataFrame([{"date": "2024-01-02", "close": 1}]).to_csv(primary / "0050_TW.csv", index=False)
            pd.DataFrame([{"date": "2024-01-02", "close": 1}]).to_csv(primary / "00631L_TW.csv", index=False)
            repair = root / "backtest_cache" / "ad_hoc_2020"
            repair.mkdir()
            pd.DataFrame([{"date": "2020-01-02", "close": 1}]).to_csv(repair / "0050_TW.csv", index=False)

            manifest = run_benchmark_cache_coverage_audit(
                repo_root=root,
                context_dir=context,
                output_dir=root / "out",
            )

            self.assertEqual(manifest["status"], "completed_benchmark_cache_coverage_audit")
            self.assertFalse(manifest["formal_model_changed"])
            missing = pd.read_csv(root / "out" / "benchmark_missing_months.csv")
            self.assertIn("2020-01", set(missing["candidate_month"]))
            candidates = pd.read_csv(root / "out" / "join_repair_candidates.csv")
            self.assertGreaterEqual(len(candidates), 1)
            saved = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["portfolio_replay_executed"])


if __name__ == "__main__":
    unittest.main()
