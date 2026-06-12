from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.radar_snapshot_readiness import REQUIRED_SNAPSHOT_COLUMNS
from backtest_lab.stock_pool_observation import (
    build_stock_pool_observation,
    run_stock_pool_observation_batch,
    write_stock_pool_observation,
)
from backtest_lab.stock_pool_store import symbol_entry


class StockPoolObservationTest(unittest.TestCase):
    def test_build_observation_outputs_unified_schema_and_top_candidate(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [
                symbol_entry("2330.TW", source="manual"),
                symbol_entry("2454.TW", source="manual"),
            ],
        }
        prices = {
            "2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000),
            "2454.TW": _trend_frame(dates, start=100, step=0.8, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )

        self.assertEqual(observation.schema_version, 1)
        self.assertEqual(observation.pool_id, "custom_ai_pool")
        self.assertEqual(observation.signal_date, dates[-1].strftime("%Y-%m-%d"))
        self.assertEqual(observation.candidate_count, 2)
        self.assertEqual(observation.action_state, "watch_candidate")
        self.assertEqual(observation.top_ticker, "2454.TW")
        self.assertEqual(observation.top_display, "聯發科(2454)")
        self.assertGreaterEqual(observation.passed_count, 1)

    def test_observation_resolves_to_previous_common_trading_date(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        )

        self.assertEqual(observation.signal_date, dates[-1].strftime("%Y-%m-%d"))
        self.assertEqual(observation.data_end_date, dates[-1].strftime("%Y-%m-%d"))

    def test_observation_can_require_exact_signal_date(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}

        with self.assertRaisesRegex(ValueError, "No exact common price data"):
            build_stock_pool_observation(
                pool=pool,
                prices_by_ticker=prices,
                signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                require_exact_signal_date=True,
            )

    def test_write_observation_outputs_json_and_candidates_csv(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker={"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)},
            signal_date=dates[-1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_stock_pool_observation(output_dir, observation)

            self.assertTrue((output_dir / "stock_pool_observation.json").exists())
            self.assertTrue((output_dir / "stock_pool_observation_candidates.csv").exists())

    def test_batch_writes_manifest_and_skips_empty_pool(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        source_cache = {
            "2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000),
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            source_cache["2330.TW"].reset_index(names="date").to_csv(cache_dir / "2330_TW.csv", index=False)
            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "custom_ai_pool",
                        "name": "自訂AI觀察池",
                        "strategy_preset": "universal_pool_custom",
                        "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
                    },
                    {
                        "pool_id": "empty_radar_pool",
                        "name": "空雷達池",
                        "strategy_preset": "radar_core_mid_small_calibrated_v1",
                        "resolved_symbols": [],
                    },
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=Path(tmp) / "out",
            )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(len(manifest["skipped"]), 1)
            self.assertEqual(manifest["skipped"][0]["reason"], "missing_radar_snapshot_dir")
            manifest_path = Path(manifest["output_root"]) / "stock_pool_observation_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_summary.csv").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_report.md").exists())
            self.assertTrue((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())
            self.assertTrue(
                (Path(manifest["generated"][0]["output_dir"]) / "stock_pool_observation.json").exists()
            )

    def test_batch_excludes_non_operational_scorecard_pool_by_default(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value=prices):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "large_cap_best_v20260605",
                            "name": "AI中大型權值股池最佳版 v20260605",
                            "strategy_preset": "best_v20260605",
                            "operational_observation": True,
                            "resolved_symbols": [symbol_entry("2330.TW", source="fixed")],
                        },
                        {
                            "pool_id": "model_scorecard_ep10",
                            "name": "模型延遲公開成績單池",
                            "strategy_preset": "delayed_public_scorecard_v1",
                            "operational_observation": False,
                            "resolved_symbols": [symbol_entry("2330.TW", source="dynamic")],
                        },
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                )

            self.assertEqual([item["pool_id"] for item in manifest["generated"]], ["large_cap_best_v20260605"])
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertNotIn("模型延遲公開成績單池", report)

    def test_batch_resolves_radar_pool_from_snapshot_dir(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            snapshot_dir = root / "snapshots"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.5, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "1111_TW.csv",
                index=False,
            )
            _write_snapshot(snapshot_dir / "radar_snapshot_20250601.csv", date=dates[-1].strftime("%Y-%m-%d"))

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "radar_mid_small_calibrated_v1",
                        "name": "雷達中小型校準版",
                        "strategy_preset": "radar_core_mid_small_calibrated_v1",
                        "resolved_symbols": [],
                    }
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
                radar_snapshot_dir=snapshot_dir,
                radar_top_n=5,
            )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["pool_id"], "radar_mid_small_calibrated_v1")
            self.assertEqual(manifest["generated"][0]["top_ticker"], "1111.TW")
            self.assertEqual(manifest["skipped"], [])

    def test_batch_generates_pool_with_partial_price_coverage(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.5, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "1111_TW.csv",
                index=False,
            )

            def fake_download(*, tickers, **kwargs):
                if tickers == ["9999.TW"]:
                    raise ValueError("missing test price")
                return {"1111.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=fake_download):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "partial_pool",
                            "name": "部分價格覆蓋池",
                            "strategy_preset": "universal_pool_custom",
                            "resolved_symbols": [
                                {
                                    "ticker": "1111.TW",
                                    "display": "可用(1111)",
                                    "asset_type": "stock",
                                },
                                {
                                    "ticker": "9999.TW",
                                    "display": "缺價(9999)",
                                    "asset_type": "stock",
                                },
                            ],
                        }
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["top_ticker"], "1111.TW")
            self.assertEqual(manifest["generated"][0]["missing_price_tickers"], ["9999.TW"])
            self.assertEqual(manifest["skipped"], [])
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("缺價股票", report)
            self.assertIn("9999.TW", report)

    def test_batch_skips_without_latest_pdf_when_exact_signal_date_missing(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            prices = {"1111.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value=prices):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "strict_pool",
                            "name": "嚴格日期池",
                            "strategy_preset": "universal_pool_custom",
                            "resolved_symbols": [{"ticker": "1111.TW", "display": "測試(1111)", "asset_type": "stock"}],
                        }
                    ],
                    signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                    require_exact_signal_date=True,
                )

            self.assertEqual(manifest["generated"], [])
            self.assertEqual(len(manifest["skipped"]), 1)
            self.assertIn("No exact common price data", manifest["skipped"][0]["reason"])
            self.assertFalse((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())


def _trend_frame(dates: pd.DatetimeIndex, *, start: float, step: float, volume: int) -> pd.DataFrame:
    closes = [start + index * step for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [volume] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


def _write_snapshot(path: Path, *, date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_SNAPSHOT_COLUMNS))
        writer.writeheader()
        row = {column: "" for column in sorted(REQUIRED_SNAPSHOT_COLUMNS)}
        row.update(
            {
                "date": date,
                "theme": "記憶體",
                "symbol": "1111",
                "name": "測試記憶體",
                "theme_rank": "1",
                "theme_score": "90",
                "capital_share": "0.2",
                "turnover_value": "100000000",
                "stock_score": "85",
                "bucket": "theme_leader",
                "fundamental_pass": "true",
                "fundamental_score": "80",
                "fundamental_data_status": "ok",
                "fundamental_source_date": date,
                "risk_heat": "0.2",
                "liquidity": "ok",
                "stock_turnover_rank_in_theme": "1",
                "stock_turnover_share_in_theme": "0.5",
                "theme_leader_flag": "true",
                "theme_second_line_flag": "false",
                "theme_laggard_rebound_flag": "false",
                "overheated_flag": "false",
            }
        )
        writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
