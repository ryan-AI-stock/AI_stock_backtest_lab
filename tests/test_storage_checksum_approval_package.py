import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.storage_checksum_approval_package import run_storage_checksum_approval_package


class StorageChecksumApprovalPackageTest(unittest.TestCase):
    def test_builds_checksums_and_approval_table_without_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "outputs" / "audit"
            candidate = root / "outputs" / "tmp_debug_run"
            candidate.mkdir(parents=True)
            (candidate / "a.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            audit.mkdir(parents=True)
            (audit / "manifest.json").write_text(
                json.dumps({"keep_required_size_mb": 1.0, "total_scanned_size_mb": 2.0}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "path": "outputs/tmp_debug_run",
                        "governance_bucket": "disposable_rebuildable_candidate",
                        "recommended_action": "delete_or_archive_after_user_approval",
                        "reason": "debug fixture",
                        "size_bytes": 10,
                        "size_mb": 0.001,
                    }
                ]
            ).to_csv(audit / "disposable_rebuildable_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "path": "backtest_cache",
                        "size_bytes": 100,
                        "size_mb": 0.1,
                    }
                ]
            ).to_csv(audit / "keep_required_for_backtest.csv", index=False)

            output = run_storage_checksum_approval_package(repo_root=root, audit_dir=audit, output_dir=root / "outputs" / "checksum")

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["candidate_rows"], 1)
            self.assertFalse(manifest["delete_executed"])
            self.assertFalse(manifest["move_executed"])
            self.assertFalse(manifest["compress_executed"])
            self.assertFalse(manifest["archive_executed"])

            approval = pd.read_csv(output / "user_approval_table.csv")
            self.assertTrue(approval["approval_required"].astype(bool).all())
            self.assertEqual(approval["approval_status"].iloc[0], "pending_user_approval")
            self.assertNotEqual(approval["aggregate_sha256"].iloc[0], "")

            protected = pd.read_csv(output / "protected_boundaries_confirmation.csv")
            self.assertTrue(protected["protected_confirmed"].astype(bool).all())
            self.assertTrue((candidate / "a.csv").exists())


if __name__ == "__main__":
    unittest.main()
