from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    _build_pool2_panel,
    _formal_policy_input_readiness,
    _load_price_source,
    _manifest,
    FORMAL_CANDIDATE_EXCLUDED_TICKERS,
)


class CurrentFormalPoolSignalPanelTests(unittest.TestCase):
    def test_formal_policy_readiness_does_not_promote_ranking_only_pool1(self) -> None:
        dates = [pd.Timestamp("2020-08-31")]
        readiness = _formal_policy_input_readiness(
            dates,
            {
                "2020-08-31": {
                    "pool1_top_candidate": "00631L.TW",
                    "pool1_formal_vote_ready": False,
                    "pool1_blocker": "missing_pool1_formal_contract",
                }
            },
            {"2020-08-31": {"pool2_vote": "2330.TW", "pool2_vote_ready": True}},
        )

        row = readiness.iloc[0]
        self.assertEqual(row["sufficient_for_pool1_primary_pool2_confirmation"], "false")
        self.assertIn("missing_pool1_formal_contract", row["blocker_reason"])

    def test_pool2_panel_filters_0050_from_candidate_rows(self) -> None:
        signal_date = pd.Timestamp("2020-08-31")
        dates = pd.bdate_range(end=signal_date, periods=160)
        benchmark = _price_frame(dates, start=100.0, daily_step=0.02)
        strong_stock = _price_frame(dates, start=100.0, daily_step=0.35)
        anchor = pd.DataFrame(
            [
                {
                    "effective_month": "2020-08",
                    "effective_date": signal_date,
                    "ticker": "0050",
                    "name": "0050",
                    "source_type": "source_backed_manual_candidate",
                    "formal_exact": "false",
                },
                {
                    "effective_month": "2020-08",
                    "effective_date": signal_date,
                    "ticker": "2330",
                    "name": "台積電",
                    "source_type": "source_backed_manual_candidate",
                    "formal_exact": "false",
                },
            ]
        )

        panel, daily = _build_pool2_panel(
            trading_dates=[signal_date],
            anchor=anchor,
            prices_by_ticker={"0050.TW": benchmark, "2330.TW": strong_stock},
            names={"2330.TW": "台積電", "0050.TW": "0050"},
            price_source_meta={"2330.TW": {"adjusted_close_available": True}},
        )

        self.assertNotIn("0050.TW", set(panel["candidate_ticker"]))
        self.assertIn("0050.TW", FORMAL_CANDIDATE_EXCLUDED_TICKERS)
        self.assertEqual(daily["2020-08-31"]["pool2_vote"], "2330.TW")
        row = panel.loc[panel["candidate_ticker"].eq("2330.TW")].iloc[0]
        self.assertEqual(row["persistence_passed"], "true")
        self.assertEqual(row["eligible_for_pool_selection"], "true")

    def test_supplemental_price_source_overrides_same_date_base_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2015-03-19",
                        "open": 0.83,
                        "high": 0.84,
                        "low": 0.82,
                        "close": 0.83,
                        "adj_close": 0.83,
                        "volume": 1,
                    }
                ]
            ).to_csv(cache / "00631L_TW.csv", index=False)
            supplemental = root / "00631L_supplemental.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2015-03-19",
                        "open": 23.30,
                        "high": 23.60,
                        "low": 23.30,
                        "close": 23.57,
                        "adj_close": 23.57,
                        "volume": 1,
                    }
                ]
            ).to_csv(supplemental, index=False)
            registry = pd.DataFrame(
                [
                    {
                        "ticker": "00631L.TW",
                        "source_path": str(supplemental),
                        "source_type": "twse_stock_day_backfill",
                    }
                ]
            )

            frame, meta = _load_price_source("00631L.TW", price_cache_dir=cache, registry=registry)

        self.assertIsNotNone(frame)
        self.assertEqual(float(frame.loc[pd.Timestamp("2015-03-19"), "open"]), 23.30)
        self.assertTrue(meta["supplemental_used"])

    def test_manifest_keeps_report_only_flags_false(self) -> None:
        manifest = _manifest(
            pool1_panel=pd.DataFrame([{"date": "2020-08-31"}]),
            pool2_panel=pd.DataFrame([{"date": "2020-08-31"}]),
            readiness=pd.DataFrame([{"sufficient_for_pool1_primary_pool2_confirmation": "false"}]),
            blockers=pd.DataFrame([{"blocks_formal_target_stream": True}]),
            start_date=pd.Timestamp("2020-08-31"),
            end_date=pd.Timestamp("2020-08-31"),
            trading_dates=[pd.Timestamp("2020-08-31")],
            monthly_anchor_path="anchor.csv",
        )

        self.assertFalse(manifest["formal_model_changed"])
        self.assertFalse(manifest["trade_decision_changed"])
        self.assertFalse(manifest["active_in_trade_decision"])
        self.assertFalse(manifest["formal_target_stream_ready"])


def _price_frame(dates: pd.DatetimeIndex, *, start: float, daily_step: float) -> pd.DataFrame:
    values = [start + idx * daily_step for idx in range(len(dates))]
    return pd.DataFrame(
        {
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "adj_close": values,
            "volume": [20_000_000] * len(dates),
        },
        index=dates.normalize(),
    )


if __name__ == "__main__":
    unittest.main()
