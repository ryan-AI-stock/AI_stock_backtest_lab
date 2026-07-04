import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.data_governance_storage_audit import run_data_governance_storage_audit


class DataGovernanceStorageAuditTest(unittest.TestCase):
    def test_audit_classifies_storage_without_deleting_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "backtest_cache" / "prices" / "00631L_TW.csv", "date,close\n2024-01-01,10\n")
            _write(root / "outputs" / "combined_formal_target_stream_20150128_20211230_20260702" / "combined_formal_target_stream.csv", "signal_date,formal_target\n2024-01-01,00631L.TW\n")
            _write(root / "outputs" / "dynamic_pool1_taxonomy_evidence_panel_20260704" / "manifest.json", "{}")
            _write(root / "outputs" / "tmp_debug_dynamic_full" / "debug.csv", "x\n1\n")
            _write(root / "outputs" / "radar_source_archive" / "raw_sources" / "raw.json", "{}")

            output = run_data_governance_storage_audit(repo_root=root, output_dir=root / "outputs" / "audit")

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["delete_executed"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])

            keep = pd.read_csv(output / "keep_required_for_backtest.csv")
            self.assertTrue(keep["path"].str.contains("backtest_cache").any())
            self.assertTrue(keep["path"].str.contains("combined_formal_target_stream").any())

            disposable = pd.read_csv(output / "disposable_rebuildable_candidates.csv")
            self.assertTrue(disposable["path"].str.contains("tmp_debug").any())
            self.assertTrue(disposable["requires_user_approval_before_action"].astype(bool).all())

            archive = pd.read_csv(output / "archive_or_compress_candidates.csv")
            self.assertTrue(archive["path"].str.contains("raw_sources|radar_source_archive", regex=True).any())

            approval = pd.read_csv(output / "deletion_requires_user_approval.csv")
            self.assertTrue(len(approval) >= len(disposable))

            self.assertTrue((root / "outputs" / "tmp_debug_dynamic_full" / "debug.csv").exists())


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
