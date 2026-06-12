from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.sector_dynamic_pool_backtest import (
    RadarSnapshotPoolVariant,
    SectorPoolVariant,
    load_theme_members,
    score_candidates,
    simulate_radar_snapshot_pool,
    target_weights_from_snapshot_ranked,
    simulate_sector_pool,
    target_weights_from_scores,
)
from backtest_lab.radar_snapshot_v2_policy_sweep import load_date_aware_membership_symbols


class SectorDynamicPoolBacktestTest(unittest.TestCase):
    def test_load_theme_members_uses_primary_theme_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theme_map.csv"
            pd.DataFrame(
                [
                    {"theme": "記憶體", "symbol": "2408", "name": "南亞科", "role": "DRAM", "conviction": "high", "primary": "yes"},
                    {"theme": "記憶體", "symbol": "9999", "name": "排除", "role": "", "conviction": "low", "primary": "no"},
                ]
            ).to_csv(path, index=False)
            members = load_theme_members(path, "記憶體")
        self.assertEqual([member.ticker for member in members], ["2408.TW"])

    def test_target_weights_keep_single_stock_cap_and_cash(self) -> None:
        weights = target_weights_from_scores(
            [("A.TW", 1.0), ("B.TW", 0.8), ("C.TW", 0.7)],
            SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.30),
        )
        self.assertEqual(weights, {"A.TW": 0.30, "B.TW": 0.30, "C.TW": 0.30})
        self.assertLess(sum(weights.values()), 1.0)

    def test_target_weights_from_snapshot_ranked_keep_cash(self) -> None:
        weights = target_weights_from_snapshot_ranked(
            [{"ticker": "A.TW"}, {"ticker": "B.TW"}],
            RadarSnapshotPoolVariant("v2", "V2", top_n=2, max_single_weight=0.40),
        )
        self.assertEqual(weights, {"A.TW": 0.40, "B.TW": 0.40})
        self.assertLess(sum(weights.values()), 1.0)

    def test_score_candidates_uses_signal_date_not_future_prices(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=160)
        base = pd.Series(range(100, 260), index=dates, dtype=float)
        future_jump = base.copy()
        future_jump.iloc[-1] = 1000.0
        prices = {
            "A.TW": _price_frame(base),
            "B.TW": _price_frame(future_jump),
        }
        before_jump_scores = score_candidates(prices, dates[-2], SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.3, min_avg_turnover_twd=1))
        after_jump_scores = score_candidates(prices, dates[-1], SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.3, min_avg_turnover_twd=1))
        self.assertEqual(before_jump_scores[0][0], "B.TW")
        self.assertNotEqual(before_jump_scores[0][1], after_jump_scores[0][1])

    def test_simulate_sector_pool_produces_multiholding_equity(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=220)
        prices = {
            "A.TW": _price_frame(pd.Series(range(100, 320), index=dates, dtype=float)),
            "B.TW": _price_frame(pd.Series(range(90, 310), index=dates, dtype=float)),
            "C.TW": _price_frame(pd.Series(range(80, 300), index=dates, dtype=float)),
        }
        result = simulate_sector_pool(
            variant=SectorPoolVariant("v", "V", top_n=3, max_single_weight=0.30, min_avg_turnover_twd=1),
            prices_by_ticker=prices,
            labels={ticker: ticker for ticker in prices},
            asset_types={ticker: "stock" for ticker in prices},
            start_date="2023-08-01",
            end_date="2023-10-31",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )
        self.assertGreater(result.result.final_value, 0)
        self.assertIn("market_exposure", result.result.equity_curve.columns)
        self.assertFalse(result.holdings.empty)

    def test_simulate_radar_snapshot_pool_uses_snapshot_candidates(self) -> None:
        dates = pd.bdate_range("2026-05-01", periods=20)
        prices = {
            "2408.TW": _price_frame(pd.Series(range(100, 120), index=dates, dtype=float)),
            "3037.TW": _price_frame(pd.Series(range(90, 110), index=dates, dtype=float)),
        }
        snapshot_history = pd.DataFrame(
            [
                _snapshot_row("2026-05-01", "2408", "南亞科", "記憶體", 90, 80, 100),
                _snapshot_row("2026-05-01", "3037", "欣興", "PCB/載板", 80, 70, 70),
            ]
        )
        snapshot_history["date"] = pd.to_datetime(snapshot_history["date"])
        snapshot_history["fundamental_source_date"] = pd.to_datetime(snapshot_history["fundamental_source_date"])

        result = simulate_radar_snapshot_pool(
            variant=RadarSnapshotPoolVariant("v2", "V2", top_n=2, max_single_weight=0.40),
            snapshot_history=snapshot_history,
            prices_by_ticker=prices,
            symbol_to_ticker={"2408": "2408.TW", "3037": "3037.TW"},
            labels={"2408.TW": "南亞科", "3037.TW": "欣興"},
            asset_types={"2408.TW": "stock", "3037.TW": "stock"},
            start_date="2026-05-04",
            end_date="2026-05-20",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )

        self.assertGreater(result.result.final_value, 0)
        self.assertFalse(result.holdings.empty)
        self.assertIn("snapshot_date", result.score_log.columns)

    def test_load_date_aware_membership_symbols_requires_usable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "date_aware.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "2408",
                        "effective_start": "2021-10-08",
                        "source_date": "2021-10-08",
                        "source_url": "https://example.com/2408",
                        "confidence": "medium",
                    },
                    {
                        "symbol": "8088",
                        "effective_start": "",
                        "source_date": "",
                        "source_url": "",
                        "confidence": "",
                    },
                ]
            ).to_csv(path, index=False)

            self.assertEqual(load_date_aware_membership_symbols(path), {"2408"})

    def test_radar_snapshot_pool_can_hold_when_candidates_temporarily_empty(self) -> None:
        dates = pd.bdate_range("2026-05-01", periods=8)
        prices = {"2408.TW": _price_frame(pd.Series(range(100, 108), index=dates, dtype=float))}
        snapshot_history = pd.DataFrame(
            [
                _snapshot_row("2026-05-01", "2408", "南亞科", "記憶體", 90, 80, 100),
                _snapshot_row("2026-05-04", "2408", "南亞科", "記憶體", 90, 80, 100, fundamental_pass=False),
            ]
        )
        snapshot_history["date"] = pd.to_datetime(snapshot_history["date"])
        snapshot_history["fundamental_source_date"] = pd.to_datetime(snapshot_history["fundamental_source_date"])

        result = simulate_radar_snapshot_pool(
            variant=RadarSnapshotPoolVariant(
                "v2_hold",
                "V2 hold",
                top_n=1,
                max_single_weight=0.80,
                rebalance_frequency="daily",
                empty_candidate_policy="hold",
            ),
            snapshot_history=snapshot_history,
            prices_by_ticker=prices,
            symbol_to_ticker={"2408": "2408.TW"},
            labels={"2408.TW": "南亞科"},
            asset_types={"2408.TW": "stock"},
            start_date="2026-05-04",
            end_date="2026-05-08",
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
        )

        self.assertIn("2408.TW", result.result.equity_curve["current_ticker"].iloc[-1])


def _price_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
            "dividend": 0.0,
        },
        index=close.index,
    )


def _snapshot_row(
    date: str,
    symbol: str,
    name: str,
    theme: str,
    theme_score: float,
    stock_score: float,
    fundamental_score: float,
    fundamental_pass: bool = True,
) -> dict[str, object]:
    return {
        "date": date,
        "theme": theme,
        "symbol": symbol,
        "name": name,
        "theme_rank": 1,
        "theme_score": theme_score,
        "capital_share": 0.1,
        "turnover_value": 1_000_000_000,
        "stock_score": stock_score,
        "bucket": "theme_leader",
        "fundamental_pass": fundamental_pass,
        "fundamental_score": fundamental_score,
        "fundamental_data_status": "ok" if fundamental_pass else "low_quality",
        "fundamental_source_date": date,
        "risk_heat": 0.2,
        "liquidity": "ok",
        "stock_turnover_rank_in_theme": 1,
        "stock_turnover_share_in_theme": 0.5,
        "theme_leader_flag": True,
        "theme_second_line_flag": False,
        "theme_laggard_rebound_flag": False,
        "overheated_flag": False,
    }


if __name__ == "__main__":
    unittest.main()
