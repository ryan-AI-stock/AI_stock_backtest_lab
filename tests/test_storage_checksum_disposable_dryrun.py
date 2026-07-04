import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.storage_checksum_disposable_dryrun import run_storage_disposable_dryrun


class StorageChecksumDisposableDryrunTest(unittest.TestCase):
    def test_writes_noop_dryrun_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            checksum = root / "checksum"
            audit.mkdir()
            checksum.mkdir()
            (audit / "manifest.json").write_text(
                json.dumps(
                    {
                        "total_scanned_size_mb": 587.368,
                        "keep_required_size_mb": 379.450,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "path": "tmp/foo.csv",
                        "governance_bucket": "disposable_rebuildable_candidate",
                        "recommended_action": "delete_or_archive_after_user_approval",
                        "reason": "rebuildable",
                        "size_bytes": 100,
                        "size_mb": 0.001,
                    }
                ]
            ).to_csv(audit / "disposable_rebuildable_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "path": "tmp/foo.csv",
                        "aggregate_sha256": "abc",
                        "checksummed_file_count": 1,
                        "checksummed_size_bytes": 100,
                        "checksum_skipped_files": 0,
                    }
                ]
            ).to_csv(checksum / "disposable_candidate_checksum_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "path": "tmp/foo.csv",
                        "size_bytes": 100,
                        "size_mb": 0.001,
                        "classification": "disposable_rebuildable_candidate",
                        "approval_required": True,
                    }
                ]
            ).to_csv(checksum / "user_approval_table.csv", index=False)

            manifest = run_storage_disposable_dryrun(
                repo_root=root,
                audit_dir=audit,
                checksum_dir=checksum,
                output_dir=root / "out",
            )

            self.assertEqual(manifest["status"], "completed_noop_disposable_dryrun")
            self.assertFalse(manifest["delete_executed"])
            self.assertTrue((root / "out" / "handoff_number_reconciliation.md").exists())
            approval = pd.read_csv(root / "out" / "requires_user_approval.csv")
            self.assertEqual(len(approval), 1)
            self.assertTrue(bool(approval.loc[0, "approval_required"]))


if __name__ == "__main__":
    unittest.main()
