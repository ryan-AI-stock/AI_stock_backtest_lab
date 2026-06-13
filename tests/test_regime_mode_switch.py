from __future__ import annotations

import unittest

import pandas as pd

from backtest_lab.regime_mode_switch import (
    _market_risk_off,
    _has_adverse_open_gap,
    _early_rebalance_signal,
    _is_last_trading_day_of_week,
    _update_attack_gate_state,
    _update_latch_release_state,
    _update_health_gate_state,
    _update_risk_off_state,
    attack_gate_leveraged_fallback_variants,
    asymmetric_strategy_selector_variants,
    stop_latched_strategy_selector_variants,
    fast_risk_strategy_selector_variants,
    fallback_only_risk_selector_variants,
    two_stage_attack_selector_variants,
    two_stage_fast_guard_variants,
    two_stage_cash_guard_variants,
    cycle_proven_selector_variants,
    cycle_proven_robustness_variants,
    cycle_proven_history_init_variants,
    cycle_proven_preproof_exposure_variants,
    cycle_proven_cadence_variants,
    cycle_proven_adaptive_cadence_variants,
    cycle_proven_adaptive_cadence_breakout_variants,
    cycle_proven_mature_bull_cadence_variants,
    cycle_proven_asset_role_variants,
    cycle_proven_market_exposure_ladder_variants,
)


class RegimeModeSwitchTest(unittest.TestCase):
    def test_fast_risk_filter_detects_short_term_weakness(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=140)
        values = [100.0] * 110 + [100.0 - index for index in range(30)]
        prices = pd.DataFrame({"adj_close": values}, index=dates)

        self.assertTrue(_market_risk_off(prices, dates[-1], "ma60_ret20"))
        self.assertTrue(_market_risk_off(prices, dates[-1], "risk_2of3"))
        self.assertTrue(_market_risk_off(prices, dates[-1], "ma20_ret10"))
        self.assertTrue(_market_risk_off(prices, dates[-1], "ret20_dd5"))

    def test_fast_risk_filter_stays_off_in_rising_market(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=140)
        values = [100.0 + index * 0.2 for index in range(140)]
        prices = pd.DataFrame({"adj_close": values}, index=dates)

        self.assertFalse(_market_risk_off(prices, dates[-1], "ma60_ret20"))
        self.assertFalse(_market_risk_off(prices, dates[-1], "risk_2of3"))

    def test_risk_off_entry_is_immediate_and_exit_can_require_confirmation(self) -> None:
        active, streak = _update_risk_off_state(
            active=False,
            clear_streak=0,
            raw_risk_off=True,
            exit_confirmation_days=3,
        )
        self.assertTrue(active)
        self.assertEqual(streak, 0)

        for expected_streak in (1, 2):
            active, streak = _update_risk_off_state(
                active=active,
                clear_streak=streak,
                raw_risk_off=False,
                exit_confirmation_days=3,
            )
            self.assertTrue(active)
            self.assertEqual(streak, expected_streak)

        active, streak = _update_risk_off_state(
            active=active,
            clear_streak=streak,
            raw_risk_off=False,
            exit_confirmation_days=3,
        )
        self.assertFalse(active)
        self.assertEqual(streak, 0)

    def test_stop_latch_releases_only_after_confirmation(self) -> None:
        active = True
        streak = 0
        for expected_streak in (1, 2):
            active, streak = _update_latch_release_state(
                active=active,
                release_streak=streak,
                raw_release=True,
                release_confirmation_days=3,
            )
            self.assertTrue(active)
            self.assertEqual(streak, expected_streak)

        active, streak = _update_latch_release_state(
            active=active,
            release_streak=streak,
            raw_release=True,
            release_confirmation_days=3,
        )
        self.assertFalse(active)
        self.assertEqual(streak, 0)

    def test_attack_gate_latches_after_confirmation_until_explicit_reset(self) -> None:
        active, streak = _update_attack_gate_state(
            active=False,
            activation_streak=0,
            raw_activation=True,
            activation_confirmation_days=2,
        )
        self.assertFalse(active)
        self.assertEqual(streak, 1)
        active, streak = _update_attack_gate_state(
            active=active,
            activation_streak=streak,
            raw_activation=True,
            activation_confirmation_days=2,
        )
        self.assertTrue(active)
        self.assertEqual(streak, 0)
        active, streak = _update_attack_gate_state(
            active=active,
            activation_streak=streak,
            raw_activation=False,
            activation_confirmation_days=2,
        )
        self.assertTrue(active)

    def test_health_gate_failure_is_immediate_and_recovery_can_require_confirmation(self) -> None:
        active, streak = _update_health_gate_state(
            active=True,
            recovery_streak=0,
            raw_healthy=False,
            recovery_confirmation_days=3,
        )
        self.assertFalse(active)
        self.assertEqual(streak, 0)

        for expected_streak in (1, 2):
            active, streak = _update_health_gate_state(
                active=active,
                recovery_streak=streak,
                raw_healthy=True,
                recovery_confirmation_days=3,
            )
            self.assertFalse(active)
            self.assertEqual(streak, expected_streak)

        active, streak = _update_health_gate_state(
            active=active,
            recovery_streak=streak,
            raw_healthy=True,
            recovery_confirmation_days=3,
        )
        self.assertTrue(active)
        self.assertEqual(streak, 0)

    def test_leveraged_fallback_selector_compares_leader_with_0050(self) -> None:
        variants = attack_gate_leveraged_fallback_variants()

        self.assertEqual(len(variants), 12)
        self.assertTrue(all(variant.attack_gate_fallback_ticker == "0050.TW" for variant in variants))
        self.assertTrue(all(variant.defense_anchor_ticker == "00631L.TW" for variant in variants))
        self.assertEqual({variant.attack_gate_defense_rule for variant in variants}, {"ma60", "ma120", "ma200"})

    def test_asymmetric_selector_uses_leverage_only_in_bull_regimes(self) -> None:
        variant = asymmetric_strategy_selector_variants()[0]

        self.assertEqual(variant.defense_anchor_ticker_by_regime["strong_bull"], "00631L.TW")
        self.assertEqual(variant.defense_anchor_ticker_by_regime["recovery_bull"], "00631L.TW")
        self.assertEqual(variant.defense_anchor_ticker_by_regime["range_bound"], "0050.TW")
        self.assertEqual(variant.defense_anchor_ticker_by_regime["correction_bear"], "0050.TW")

    def test_stop_latched_selector_defends_with_0050_after_portfolio_stop(self) -> None:
        variant = stop_latched_strategy_selector_variants()[0]

        self.assertEqual(variant.defense_anchor_ticker, "00631L.TW")
        self.assertEqual(variant.attack_gate_stop_latch_ticker, "0050.TW")
        self.assertEqual(variant.attack_gate_stop_latch_rule, "ma245")

    def test_fast_risk_selector_uses_0050_for_risk_off_overlay(self) -> None:
        variant = fast_risk_strategy_selector_variants()[0]

        self.assertEqual(variant.defense_anchor_ticker, "00631L.TW")
        self.assertEqual(variant.market_risk_off_mode, "0050_defense")
        self.assertEqual(variant.market_risk_off_defense_ticker, "0050.TW")
        self.assertEqual(variant.market_risk_off_defense_rule, "ma245")

    def test_fallback_only_risk_selector_does_not_interrupt_confirmed_attack(self) -> None:
        variant = fallback_only_risk_selector_variants()[0]

        self.assertTrue(variant.market_risk_off_only_when_attack_gate_inactive)
        self.assertEqual(variant.attack_gate_min_top_days, 10)

    def test_two_stage_selector_has_strict_initial_gate_and_fast_reentry(self) -> None:
        variant = two_stage_attack_selector_variants()[0]

        self.assertEqual(variant.attack_gate_margin_over_fallback, 0.22)
        self.assertEqual(variant.attack_gate_min_top_days, 10)
        self.assertEqual(variant.attack_gate_reentry_margin_over_fallback, 0.20)
        self.assertEqual(variant.attack_gate_reentry_min_short_to_medium_momentum_ratio, 0.40)

    def test_two_stage_fast_guard_only_changes_waiting_state_risk_filter(self) -> None:
        variants = two_stage_fast_guard_variants()

        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant.market_risk_off_only_when_attack_gate_inactive for variant in variants))
        self.assertEqual(
            {variant.market_risk_off_filter for variant in variants},
            {"risk_2of3", "ma20_ret10", "ret20_dd5", "ma60_ret20"},
        )

    def test_two_stage_cash_guard_goes_fully_to_cash(self) -> None:
        variants = two_stage_cash_guard_variants()

        self.assertEqual(len(variants), 3)
        self.assertTrue(all(variant.market_risk_off_mode == "cash" for variant in variants))

    def test_cycle_proven_selector_limits_preproof_guard_to_new_cycles(self) -> None:
        variants = cycle_proven_selector_variants()

        self.assertEqual(len(variants), 2)
        self.assertTrue(all(variant.market_risk_off_only_before_first_attack_activation for variant in variants))
        self.assertEqual({variant.market_risk_off_mode for variant in variants}, {"cash", "0050_defense"})

    def test_cycle_proven_robustness_matrix_covers_parameter_neighborhood(self) -> None:
        variants = cycle_proven_robustness_variants()

        self.assertEqual(len(variants), 12)
        self.assertEqual({variant.attack_gate_margin_over_fallback for variant in variants}, {0.20, 0.22, 0.24})
        self.assertEqual({variant.attack_gate_min_top_days for variant in variants}, {8, 10})
        self.assertEqual(
            {variant.attack_gate_reentry_min_short_to_medium_momentum_ratio for variant in variants},
            {0.40, 0.60},
        )

    def test_cycle_proven_history_init_uses_only_prior_history(self) -> None:
        variants = cycle_proven_history_init_variants()

        self.assertEqual(
            {variant.attack_gate_initialize_history_days for variant in variants},
            {60, 120, 252},
        )
        self.assertTrue(all(variant.attack_gate_initialize_active_from_history for variant in variants))

    def test_cycle_proven_preproof_exposure_matrix_keeps_attack_rules_fixed(self) -> None:
        variants = cycle_proven_preproof_exposure_variants()

        self.assertEqual(
            {variant.market_risk_off_exposure for variant in variants},
            {0.0, 0.25, 0.50, 0.75},
        )
        self.assertTrue(all(variant.attack_gate_initialize_history_days == 60 for variant in variants))

    def test_cycle_proven_cadence_matrix_covers_daily_hybrid_and_full_weekly(self) -> None:
        variants = cycle_proven_cadence_variants()

        self.assertEqual(len(variants), 15)
        self.assertEqual(
            sum(
                variant.normal_rebalance_weekday is None
                and not variant.normal_rebalance_last_trading_day_of_week
                for variant in variants
            ),
            1,
        )
        self.assertEqual(
            {variant.normal_rebalance_weekday for variant in variants if variant.normal_rebalance_weekday is not None},
            {0, 1, 2, 3, 4},
        )
        self.assertEqual(
            {variant.state_evaluation_weekday for variant in variants if variant.state_evaluation_weekday is not None},
            {0, 1, 2, 3, 4},
        )
        self.assertTrue(
            all(variant.attack_selection_exclude_tickers == ("0050.TW", "00631L.TW") for variant in variants)
        )
        self.assertEqual(
            sum(variant.normal_rebalance_last_trading_day_of_week for variant in variants),
            3,
        )
        self.assertEqual(
            {
                variant.defer_rebalance_on_adverse_open_gap_pct
                for variant in variants
                if variant.defer_rebalance_on_adverse_open_gap_pct is not None
            },
            {0.02, 0.03},
        )

    def test_cycle_proven_adaptive_cadence_changes_only_rotation_timing(self) -> None:
        variants = cycle_proven_adaptive_cadence_variants()

        self.assertEqual(len(variants), 5)
        self.assertTrue(all(variant.attack_selection_exclude_tickers == ("0050.TW", "00631L.TW") for variant in variants))
        self.assertTrue(all(variant.normal_rebalance_weekday_by_regime for variant in variants))
        self.assertEqual(
            {variant.normal_rebalance_weekday_by_regime["strong_bull"] for variant in variants},
            {2},
        )
        self.assertEqual(
            {
                variant.normal_rebalance_last_trading_day_regimes
                for variant in variants
                if variant.normal_rebalance_last_trading_day_regimes
            },
            {("range_bound",)},
        )

    def test_cycle_proven_adaptive_cadence_breakout_adds_leadership_exception(self) -> None:
        variants = cycle_proven_adaptive_cadence_breakout_variants()

        self.assertEqual(len(variants), 10)
        self.assertEqual(
            {variant.early_rebalance_min_gap_over_current for variant in variants},
            {0.05, 0.10, 0.15, 0.20, 0.25},
        )
        self.assertTrue(all(variant.early_rebalance_regimes for variant in variants))

    def test_cycle_proven_mature_bull_cadence_delays_weekly_cadence_until_regime_matures(self) -> None:
        variants = cycle_proven_mature_bull_cadence_variants()

        self.assertEqual(len(variants), 8)
        self.assertEqual({variant.normal_rebalance_min_regime_streak_days for variant in variants}, {10, 20, 30})
        self.assertTrue(all(variant.normal_rebalance_weekday_by_regime for variant in variants))
        self.assertEqual(
            {variant.normal_rebalance_weekday_by_regime["strong_bull"] for variant in variants},
            {2},
        )
        self.assertEqual(
            {variant.early_rebalance_min_gap_over_current for variant in variants if variant.early_rebalance_regimes},
            {0.10, 0.15},
        )

    def test_early_rebalance_signal_requires_clear_new_leader(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=90)
        prices_by_ticker = {
            "AAA.TW": pd.DataFrame({"adj_close": [100 + index * 0.5 for index in range(90)]}, index=dates),
            "BBB.TW": pd.DataFrame({"adj_close": [100 + index * 1.5 for index in range(90)]}, index=dates),
            "CCC.TW": pd.DataFrame({"adj_close": [100 + index * 0.4 for index in range(90)]}, index=dates),
        }
        variant = cycle_proven_adaptive_cadence_breakout_variants()[0]

        self.assertTrue(
            _early_rebalance_signal(
                prices_by_ticker=prices_by_ticker,
                signal_date=dates[-1],
                current_ticker="AAA.TW",
                variant=variant,
            )
        )
        self.assertFalse(
            _early_rebalance_signal(
                prices_by_ticker=prices_by_ticker,
                signal_date=dates[-1],
                current_ticker="BBB.TW",
                variant=variant,
            )
        )

    def test_last_trading_day_of_week_handles_friday_holiday(self) -> None:
        trade_dates = list(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-12"]))

        self.assertTrue(_is_last_trading_day_of_week(trade_dates, pd.Timestamp("2026-01-08")))
        self.assertFalse(_is_last_trading_day_of_week(trade_dates, pd.Timestamp("2026-01-07")))

    def test_adverse_open_gap_guard_uses_next_open(self) -> None:
        market_prices = pd.DataFrame(
            {"adj_close": [100.0, 99.0], "open": [100.0, 97.9]},
            index=pd.to_datetime(["2026-06-12", "2026-06-15"]),
        )

        self.assertTrue(
            _has_adverse_open_gap(
                market_prices=market_prices,
                signal_date=pd.Timestamp("2026-06-12"),
                trade_date=pd.Timestamp("2026-06-15"),
                threshold_pct=0.02,
            )
        )
        self.assertFalse(
            _has_adverse_open_gap(
                market_prices=market_prices,
                signal_date=pd.Timestamp("2026-06-12"),
                trade_date=pd.Timestamp("2026-06-15"),
                threshold_pct=0.03,
            )
        )

    def test_cycle_proven_asset_roles_keep_etfs_in_universe_but_can_exclude_them_from_attack_ranking(self) -> None:
        variants = cycle_proven_asset_role_variants()

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            {variant.attack_selection_exclude_tickers for variant in variants},
            {(), ("00631L.TW",), ("0050.TW", "00631L.TW")},
        )
        self.assertEqual({variant.attack_gate_exclude_tickers for variant in variants}, {(), ("00631L.TW",)})

    def test_cycle_proven_market_exposure_ladders_keep_attack_engine_fixed(self) -> None:
        variants = cycle_proven_market_exposure_ladder_variants()

        self.assertEqual(len(variants), 5)
        self.assertTrue(all(variant.attack_selection_exclude_tickers == ("0050.TW", "00631L.TW") for variant in variants))
        self.assertTrue(all(variant.attack_gate_margin_over_fallback == 0.22 for variant in variants))
        self.assertTrue(all(variant.attack_gate_defense_exposure_by_regime is not None for variant in variants))


if __name__ == "__main__":
    unittest.main()
