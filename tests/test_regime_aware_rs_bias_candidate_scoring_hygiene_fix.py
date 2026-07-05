import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.regime_aware_rs_bias_candidate_scoring_hygiene_fix import (
    run_regime_aware_rs_bias_candidate_scoring_hygiene_fix,
)


class RegimeAwareRsBiasCandidateScoringHygieneFixTest(unittest.TestCase):
    def test_forces_case_trace_rows_to_reference_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "contract.csv"
            future = root / "future.csv"
            violation = root / "violation.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "2454.TW",
                        "candidate_month": "2026-06",
                        "as_of_date": "2026-06-30",
                        "branch_variant": "long_strong_rs40_bias_guard",
                        "case_trace_only": True,
                        "branch_candidate_label": "candidate_context",
                        "branch_candidate_selected": True,
                    },
                    {
                        "ticker": "2330.TW",
                        "candidate_month": "2026-06",
                        "as_of_date": "2026-06-30",
                        "branch_variant": "long_strong_rs40_bias_guard",
                        "case_trace_only": False,
                        "branch_candidate_label": "candidate_context",
                        "branch_candidate_selected": True,
                    },
                ]
            ).to_csv(source, index=False)
            pd.DataFrame(
                [{"audit_item": "source", "rows": 2, "future_data_violation": False, "reason": "ok"}]
            ).to_csv(future, index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "2454.TW",
                        "candidate_month": "2026-06",
                        "as_of_date": "2026-06-30",
                        "branch_variant": "long_strong_rs40_bias_guard",
                    }
                ]
            ).to_csv(violation, index=False)

            manifest = run_regime_aware_rs_bias_candidate_scoring_hygiene_fix(
                repo_root=root,
                source_contract=source,
                source_future_audit=future,
                upstream_violation=violation,
                output_dir=root / "out",
            )

            self.assertEqual(manifest["before_case_trace_selected_violation_rows"], 1)
            self.assertEqual(manifest["after_case_trace_selected_violation_rows"], 0)
            fixed = pd.read_csv(root / "out" / "regime_aware_rs_bias_candidate_contract_hygiene_fixed.csv")
            case_rows = fixed[fixed["case_trace_only"].astype(str).str.lower().eq("true")]
            self.assertFalse(bool(case_rows["branch_candidate_selected"].any()))
            self.assertEqual(set(case_rows["branch_candidate_label"]), {"case_trace_reference_only"})
            summary = pd.read_csv(root / "out" / "selected_rows_before_after_summary.csv")
            selected = summary[summary["metric"].eq("selected_rows_all")].iloc[0]
            self.assertEqual(int(selected["before_value"]), 2)
            self.assertEqual(int(selected["after_value"]), 1)


if __name__ == "__main__":
    unittest.main()
