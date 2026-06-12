from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.radar_core_pool_v1 import (
    RadarCoreVariant,
    _apply_rebalance_band,
    _blowoff_risk,
    _is_rebalance_date,
    _portfolio_stop_triggered,
    _position_stop_triggered,
    _theme_strength_adjusted_exposure,
    _theme_relative_metrics,
    _trend_exit_risk,
    radar_core_mid_small_calibrated_v1_variant,
    resolve_variant_for_pool,
    score_stocks,
    simulate_radar_core_pool,
    target_weights_from_radar_scores,
)
from backtest_lab.sector_dynamic_pool_backtest import ThemeMember


class RadarCorePoolV1Test(unittest.TestCase):
    def test_target_weights_select_by_theme_and_exposure(self) -> None:
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
            "C.TW": ThemeMember("PCB", "C.TW", "C", "C", "", "high"),
            "D.TW": ThemeMember("PCB", "D.TW", "D", "D", "", "medium"),
        }
        weights = target_weights_from_radar_scores(
            theme_scores={"記憶體": 0.4, "PCB": 0.3},
            stock_scores={"A.TW": 0.9, "B.TW": 0.8, "C.TW": 0.7, "D.TW": 0.6},
            members_by_ticker=members,
            exposure=0.6,
            variant=RadarCoreVariant(top_theme_count=2, max_stocks_per_theme=1, max_single_weight=0.25),
        )
        self.assertEqual(weights, {"A.TW": 0.25, "C.TW": 0.25})

    def test_global_leaders_selects_strongest_stock_inside_qualified_themes(self) -> None:
        members = {
            "A.TW": ThemeMember("最強題材", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("第二題材", "B.TW", "B", "B", "", "high"),
            "C.TW": ThemeMember("弱題材", "C.TW", "C", "C", "", "high"),
        }
        weights = target_weights_from_radar_scores(
            theme_scores={"最強題材": 0.4, "第二題材": 0.39, "弱題材": 0.01},
            stock_scores={"A.TW": 0.2, "B.TW": 0.9, "C.TW": 1.1},
            members_by_ticker=members,
            exposure=1.0,
            variant=RadarCoreVariant(
                top_theme_count=2,
                max_stocks_per_theme=1,
                max_single_weight=1.0,
                min_theme_score=0.12,
                selection_mode="global_leaders",
                max_total_stocks=1,
            ),
        )
        self.assertEqual(weights, {"B.TW": 1.0})

    def test_hold_existing_leader_when_score_is_close_enough(self) -> None:
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }
        weights = target_weights_from_radar_scores(
            theme_scores={"記憶體": 0.3},
            stock_scores={"A.TW": 0.9, "B.TW": 1.0},
            members_by_ticker=members,
            exposure=1.0,
            variant=RadarCoreVariant(
                top_theme_count=1,
                max_stocks_per_theme=1,
                max_single_weight=1.0,
                hold_existing_score_ratio=0.85,
            ),
            current_tickers={"A.TW"},
        )
        self.assertEqual(weights, {"A.TW": 1.0})

    def test_hold_existing_leader_switches_when_score_falls_too_far(self) -> None:
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }
        weights = target_weights_from_radar_scores(
            theme_scores={"記憶體": 0.3},
            stock_scores={"A.TW": 0.7, "B.TW": 1.0},
            members_by_ticker=members,
            exposure=1.0,
            variant=RadarCoreVariant(
                top_theme_count=1,
                max_stocks_per_theme=1,
                max_single_weight=1.0,
                hold_existing_score_ratio=0.85,
            ),
            current_tickers={"A.TW"},
        )
        self.assertEqual(weights, {"B.TW": 1.0})

    def test_low_theme_score_stays_cash(self) -> None:
        weights = target_weights_from_radar_scores(
            theme_scores={"記憶體": 0.01},
            stock_scores={"A.TW": 0.9},
            members_by_ticker={"A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high")},
            exposure=0.9,
            variant=RadarCoreVariant(min_theme_score=0.12),
        )
        self.assertEqual(weights, {})

    def test_weak_theme_strength_reduces_exposure_until_full_score(self) -> None:
        variant = RadarCoreVariant(
            min_theme_score=0.20,
            weak_theme_full_score=0.25,
            weak_theme_exposure_multiplier=0.5,
        )
        self.assertEqual(
            _theme_strength_adjusted_exposure(1.0, {"記憶體": 0.23}, variant),
            0.5,
        )
        self.assertEqual(
            _theme_strength_adjusted_exposure(1.0, {"記憶體": 0.26}, variant),
            1.0,
        )

    def test_score_stocks_filters_overheated_stock(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=150)
        steady = pd.Series(range(100, 250), index=dates, dtype=float)
        overheated = steady.copy()
        overheated.iloc[-10:] = overheated.iloc[-10:] * 2.5
        prices = {
            "A.TW": _price_frame(steady),
            "B.TW": _price_frame(overheated),
        }
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }
        scores = score_stocks(
            prices,
            members,
            dates[-1],
            RadarCoreVariant(min_avg_turnover_twd=1, overheated_20d_return=0.55),
        )
        self.assertIn("A.TW", scores)
        self.assertNotIn("B.TW", scores)

    def test_profile_defaults_choose_large_pool_settings(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=150)
        prices = {
            f"{ticker}.TW": _price_frame(pd.Series(range(100, 250), index=dates, dtype=float), volume=20_000_000)
            for ticker in ("2330", "2454", "2308", "2317", "2382")
        }
        members = {
            ticker: ThemeMember("AI大型權值", ticker, ticker.split(".")[0], ticker, "", "high")
            for ticker in prices
        }

        resolved = resolve_variant_for_pool(
            variant=RadarCoreVariant(use_pool_profile_defaults=True),
            prices_by_ticker=prices,
            members_by_ticker=members,
            signal_date=dates[-1],
        )

        self.assertEqual(resolved.min_avg_turnover_twd, 0.0)
        self.assertEqual(resolved.stock_score_mode, "relative_strength")
        self.assertAlmostEqual(resolved.overheated_20d_return, 0.90)

    def test_profile_defaults_choose_mid_small_pool_settings(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=150)
        prices = {
            f"{ticker}.TW": _price_frame(pd.Series(range(80, 230), index=dates, dtype=float), volume=800_000)
            for ticker in ("2408", "2344", "2337", "3006", "3260", "8299", "8271", "8088", "6531", "6770")
        }
        members = {
            ticker: ThemeMember("記憶體", ticker, ticker.split(".")[0], ticker, "", "high")
            for ticker in prices
        }

        resolved = resolve_variant_for_pool(
            variant=RadarCoreVariant(use_pool_profile_defaults=True),
            prices_by_ticker=prices,
            members_by_ticker=members,
            signal_date=dates[-1],
        )

        self.assertEqual(resolved.min_avg_turnover_twd, 60_000_000)
        self.assertEqual(resolved.stock_score_mode, "risk_adjusted")
        self.assertAlmostEqual(resolved.overheated_20d_return, 0.62)

    def test_mid_small_calibrated_preset_matches_research_winner(self) -> None:
        variant = radar_core_mid_small_calibrated_v1_variant()

        self.assertEqual(variant.name, "radar_core_v1_score_risk_stock00_turnover60m_overheat62")
        self.assertEqual(variant.top_theme_count, 1)
        self.assertEqual(variant.max_stocks_per_theme, 1)
        self.assertEqual(variant.max_single_weight, 1.0)
        self.assertEqual(variant.min_avg_turnover_twd, 60_000_000)
        self.assertEqual(variant.min_stock_score, 0.0)
        self.assertAlmostEqual(variant.overheated_20d_return, 0.62)
        self.assertEqual(variant.stock_score_mode, "risk_adjusted")

    def test_acceleration_score_rewards_recent_turnover_surge(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=150)
        base = pd.Series(range(100, 250), index=dates, dtype=float)
        quiet = _price_frame(base)
        surge = _price_frame(base)
        quiet["volume"] = [1_000_000] * 150
        surge["volume"] = [1_000_000] * 145 + [4_000_000] * 5
        prices = {"A.TW": quiet, "B.TW": surge}
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }

        scores = score_stocks(
            prices,
            members,
            dates[-1],
            RadarCoreVariant(min_avg_turnover_twd=1, stock_score_mode="acceleration"),
        )

        self.assertGreater(scores["B.TW"], scores["A.TW"])

    def test_theme_relative_score_rewards_leader_inside_same_theme(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=150)
        steady = pd.Series(range(100, 250), index=dates, dtype=float)
        leader = steady.copy()
        leader.iloc[-40:] = leader.iloc[-40:] * 1.35
        prices = {
            "A.TW": _price_frame(steady),
            "B.TW": _price_frame(leader),
        }
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }

        metrics = _theme_relative_metrics(prices, members, dates[-1])
        scores = score_stocks(
            prices,
            members,
            dates[-1],
            RadarCoreVariant(
                min_avg_turnover_twd=1,
                stock_score_mode="risk_adjusted_theme_relative",
                overheated_20d_return=1.0,
            ),
        )

        self.assertGreater(metrics["B.TW"]["ret60_relative"], 0)
        self.assertGreater(scores["B.TW"], scores["A.TW"])

    def test_simulation_uses_market_calendar_instead_of_common_stock_dates(self) -> None:
        market_dates = pd.bdate_range("2022-01-03", periods=360)
        full_close = pd.Series(range(100, 460), index=market_dates, dtype=float)
        late_close = pd.Series(range(100, 100 + len(market_dates[300:])), index=market_dates[300:], dtype=float)
        prices = {
            "A.TW": _price_frame(full_close),
            "B.TW": _price_frame(late_close),
        }
        members = {
            "A.TW": ThemeMember("記憶體", "A.TW", "A", "A", "", "high"),
            "B.TW": ThemeMember("記憶體", "B.TW", "B", "B", "", "high"),
        }

        result = simulate_radar_core_pool(
            name="calendar_test",
            prices_by_ticker=prices,
            members_by_ticker=members,
            asset_types={"A.TW": "stock", "B.TW": "stock"},
            market_prices=_price_frame(full_close),
            start_date=_date_text(market_dates[260]),
            end_date=_date_text(market_dates[280]),
            initial_cash=1_000_000,
            cost_model=TaiwanCostModel(),
            variant=RadarCoreVariant(min_avg_turnover_twd=1, min_theme_score=9),
        )

        self.assertEqual(result.result.equity_curve.index[0], market_dates[260])
        self.assertEqual(result.result.equity_curve.index[-1], market_dates[280])
        self.assertEqual(result.result.final_value, 1_000_000)

    def test_rebalance_band_keeps_small_weight_drift(self) -> None:
        weights = _apply_rebalance_band(
            positions={"A.TW": 2700},
            close_prices={"A.TW": 100},
            total_value=1_000_000,
            target_weights={"A.TW": 0.25},
            band=0.05,
        )
        self.assertEqual(weights, {"A.TW": 0.27})

    def test_biweekly_rebalance_skips_alternating_week_boundaries(self) -> None:
        trade_dates = list(pd.bdate_range("2023-01-02", "2023-01-23"))
        rebalance_dates = [
            date.strftime("%Y-%m-%d")
            for index, date in enumerate(trade_dates)
            if _is_rebalance_date(trade_dates, index, "biweekly")
        ]
        self.assertEqual(rebalance_dates, ["2023-01-02", "2023-01-16"])

    def test_weekly_target_weekday_rebalances_on_first_available_day(self) -> None:
        trade_dates = [
            pd.Timestamp("2023-01-02"),
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-04"),
            pd.Timestamp("2023-01-06"),
            pd.Timestamp("2023-01-09"),
            pd.Timestamp("2023-01-10"),
        ]
        rebalance_dates = [
            date.strftime("%Y-%m-%d")
            for index, date in enumerate(trade_dates)
            if _is_rebalance_date(trade_dates, index, "weekly_thu")
        ]
        self.assertEqual(rebalance_dates, ["2023-01-02", "2023-01-06"])

    def test_blowoff_risk_requires_runup_drawdown_and_volume(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=80)
        close = pd.Series([100.0] * 60 + [110, 120, 132, 145, 155, 150, 142, 135, 130, 128] + [126] * 10, index=dates)
        frame = _price_frame(close)
        frame["volume"] = [1_000_000] * 60 + [3_000_000] * 20

        self.assertTrue(
            _blowoff_risk(
                frame,
                dates[69],
                RadarCoreVariant(
                    daily_blowoff_exit=True,
                    blowoff_min_runup_20d=0.20,
                    blowoff_drawdown_10d=-0.08,
                    blowoff_volume_ratio_5d_over_60d=1.2,
                ),
            )
        )

    def test_blowoff_risk_ignores_orderly_uptrend(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=80)
        close = pd.Series(range(100, 180), index=dates, dtype=float)
        frame = _price_frame(close)
        frame["volume"] = [1_000_000] * 80

        self.assertFalse(
            _blowoff_risk(
                frame,
                dates[-1],
                RadarCoreVariant(daily_blowoff_exit=True),
            )
        )

    def test_trend_exit_risk_triggers_when_price_breaks_short_trend(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=40)
        close = pd.Series([100.0 + i for i in range(30)] + [120, 116, 112, 108, 105, 103, 101, 99, 98, 97], index=dates)
        self.assertTrue(
            _trend_exit_risk(
                _price_frame(close),
                dates[-1],
                RadarCoreVariant(daily_trend_exit=True, trend_exit_ma_window=10),
            )
        )

    def test_trend_exit_risk_ignores_healthy_uptrend(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=40)
        close = pd.Series(range(100, 140), index=dates, dtype=float)
        self.assertFalse(
            _trend_exit_risk(
                _price_frame(close),
                dates[-1],
                RadarCoreVariant(daily_trend_exit=True, trend_exit_ma_window=10),
            )
        )

    def test_portfolio_stop_triggers_only_after_configured_drawdown(self) -> None:
        variant = RadarCoreVariant(portfolio_trailing_stop=True, portfolio_stop_drawdown=-0.12)
        self.assertFalse(_portfolio_stop_triggered(900_000, 1_000_000, variant))
        self.assertTrue(_portfolio_stop_triggered(879_000, 1_000_000, variant))

    def test_portfolio_stop_can_wait_until_min_gain(self) -> None:
        variant = RadarCoreVariant(
            portfolio_trailing_stop=True,
            portfolio_stop_drawdown=-0.12,
            portfolio_stop_min_gain=2.0,
        )
        self.assertFalse(_portfolio_stop_triggered(1_700_000, 1_900_000, variant, 1_000_000))
        self.assertTrue(_portfolio_stop_triggered(2_700_000, 3_100_000, variant, 1_000_000))

    def test_position_stop_waits_for_runup_then_triggers_on_pullback(self) -> None:
        variant = RadarCoreVariant(
            position_trailing_stop=True,
            position_stop_min_runup=0.30,
            position_stop_drawdown=-0.15,
        )
        self.assertFalse(
            _position_stop_triggered(signal_price=110, entry_price=100, peak_price=125, variant=variant)
        )
        self.assertFalse(
            _position_stop_triggered(signal_price=120, entry_price=100, peak_price=140, variant=variant)
        )
        self.assertTrue(
            _position_stop_triggered(signal_price=118, entry_price=100, peak_price=140, variant=variant)
        )


def _price_frame(close: pd.Series, *, volume: int = 1_000_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=close.index,
    )


def _date_text(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%d")


if __name__ == "__main__":
    unittest.main()
