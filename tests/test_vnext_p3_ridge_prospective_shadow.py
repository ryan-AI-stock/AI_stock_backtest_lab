import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab import vnext_p3_ridge_prospective_shadow as shadow


class ProspectiveShadowTest(unittest.TestCase):
    def test_closed_market_skips(self):
        self.assertEqual(shadow.validate_manifest({"market_status": "market_closed_no_signal"}, "2026-07-12"), "market_closed_no_signal")

    def test_carried_scope_is_rejected(self):
        manifest = {"decision_date": "2026-07-13", "market_status": "open", "data_ready": True, "exact_layer4_primary80": True, "candidate_count": 80, "candidate_scope_semantics": "carry_forward", "available_at": "x", "candidate_universe_version": "v", "candidate_universe_hash": "h", "corporate_action_guard_status": "ready"}
        with self.assertRaisesRegex(RuntimeError, "carried"):
            shadow.validate_manifest(manifest, "2026-07-13")

    def test_atomic_append_refuses_changed_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.csv"
            shadow.atomic_write(path, "a\n")
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(original, shadow.sha256(path))

    def test_outcome_evaluator_does_not_modify_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pred = root / "pred/2026/07/13"
            pred.mkdir(parents=True)
            pred_file = pred / "p3_ridge_shadow_predictions.csv"
            pd.DataFrame([{"shadow_id": "R0", "decision_date": "2026-07-13", "ticker": "2330"}]).to_csv(pred_file, index=False)
            before = shadow.sha256(pred_file)
            outcome = root / "outcome.csv"
            pd.DataFrame([{"decision_date": "2026-07-13", "ticker": "2330", "horizon_td": 20, "outcome_status": "mature_official", "actual_net_excess_vs_0050": 0.01, "path_MDD": -0.02, "tail_daily_return_p10": -0.01, "large_down_count": 0, "outcome_available_at": "2026-08-10T14:00:00+08:00"}]).to_csv(outcome, index=False)
            result = shadow.evaluate_matured(root / "pred", outcome, root / "eval")
            self.assertEqual(result["evaluated_rows"], 1)
            self.assertEqual(before, shadow.sha256(pred_file))


if __name__ == "__main__":
    unittest.main()
