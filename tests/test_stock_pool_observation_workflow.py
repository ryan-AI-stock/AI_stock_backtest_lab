from __future__ import annotations

import unittest
from pathlib import Path


class StockPoolObservationWorkflowTest(unittest.TestCase):
    def test_workflow_checks_out_radar_repo_and_passes_formal_radar_data_dir(self) -> None:
        workflow = Path(".github/workflows/stock_pool_observation.yml").read_text(encoding="utf-8")

        self.assertIn("repository: ryan-AI-stock/AI_stock_rotation_radar", workflow)
        self.assertIn("path: AI_stock_rotation_radar", workflow)
        self.assertIn("AI_stock_rotation_radar/data", workflow)
        self.assertIn("RADAR_DATA_DIR", workflow)
        self.assertIn("--radar-data-dir", workflow)
        self.assertIn("MARKET_CAP_DATA_PATH", workflow)
        self.assertIn("--market-cap-data", workflow)
        self.assertIn("stock_metrics.refreshed.csv", workflow)
        self.assertIn("INSTITUTIONAL_FLOW_DATA_PATH", workflow)
        self.assertIn("--institutional-flow-data", workflow)
        self.assertIn("MARGIN_SHORT_DATA_PATH", workflow)
        self.assertIn("--margin-short-data", workflow)
        self.assertIn("BORROW_LENDING_DATA_PATH", workflow)
        self.assertIn("--borrow-lending-data", workflow)
        self.assertIn("DAY_TRADING_DATA_PATH", workflow)
        self.assertIn("--day-trading-data", workflow)
        self.assertIn("SENTIMENT_DATA_PATH", workflow)
        self.assertIn("--sentiment-data", workflow)
        self.assertIn("python -m backtest_lab.risk_factor_readiness", workflow)
        self.assertIn("risk_factor_readiness.json", workflow)
        self.assertIn("risk_factor_readiness_signals.csv", workflow)
        self.assertIn("AI_stock_rotation_radar/data/history", workflow)
        self.assertIn("RADAR_SNAPSHOT_DIR", workflow)
        self.assertIn("python -m backtest_lab.drive_publish", workflow)
        self.assertIn("AI股票池觀察總覽_最新版_v20260612.pdf", workflow)
        self.assertIn("STOCK_POOL_OBSERVATION_DRIVE_FOLDER_ID", workflow)
        self.assertIn("require_exact_signal_date:", workflow)
        self.assertIn("Manual only: require exact requested signal date", workflow)
        self.assertIn("EVENT_NAME: ${{ github.event_name }}", workflow)
        self.assertIn('echo "require_exact_signal_date=false" >> "$GITHUB_OUTPUT"', workflow)
        self.assertIn('echo "require_exact_signal_date=true" >> "$GITHUB_OUTPUT"', workflow)
        self.assertIn('exact_args+=(--require-exact-signal-date)', workflow)
        self.assertIn('"${exact_args[@]}"', workflow)
        self.assertIn("steps.observation-pdf.outputs.exists == 'true'", workflow)
        self.assertIn("STOCK_POOLS_JSON", workflow)
        self.assertIn("work/stock_pools/stock_pools.json", workflow)


if __name__ == "__main__":
    unittest.main()
