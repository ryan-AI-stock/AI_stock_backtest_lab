import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.shared_normalized_directory_migration_spec import run_shared_normalized_directory_migration_spec


class SharedNormalizedDirectoryMigrationSpecTest(unittest.TestCase):
    def test_writes_manifest_only_migration_spec_without_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = run_shared_normalized_directory_migration_spec(output_dir=Path(tmp) / "spec")

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_manifest_only_migration_spec")
            self.assertFalse(manifest["delete_executed"])
            self.assertFalse(manifest["move_executed"])
            self.assertFalse(manifest["compress_executed"])
            self.assertFalse(manifest["archive_executed"])
            self.assertTrue(manifest["requires_radar_data_next_step"])

            phases = pd.read_csv(output / "migration_phases.csv")
            self.assertIn("phase0_manifest_only", set(phases["phase"]))
            self.assertIn("phase3_user_approved_move_or_archive", set(phases["phase"]))

            datasets = pd.read_csv(output / "dataset_migration_candidates.csv")
            self.assertIn("all_listed_liquid_universe_pit_daily", set(datasets["dataset_id"]))
            self.assertIn("formal_next_day_ledgers", set(datasets["dataset_id"]))


if __name__ == "__main__":
    unittest.main()
