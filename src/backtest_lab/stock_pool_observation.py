from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices, load_price_csv
from backtest_lab.decision_layers import (
    CANDIDATE_SOURCE,
    FORMAL_TRADE_SIGNAL,
    DATA_READINESS,
    default_stock_pool_model_layer_audit,
    write_model_layer_audit,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET, formal_model_report_description
from backtest_lab.formal_radar_candidates import formal_radar_candidates_to_symbols, load_formal_radar_candidates
from backtest_lab.frozen_market_data import fill_signal_date_from_twse, incomplete_tickers, write_price_cache
from backtest_lab.market_cap_source import load_first_available_market_caps
from backtest_lab.risk_factor_source import RiskFactorSignal, load_first_available_risk_factors
from backtest_lab.frozen_report_pdf import _configure_chinese_font, _save_figure_as_raster_pdf_page
from backtest_lab.frozen_strategy_monitor import (
    AI_THEME_STRATEGY_ID,
    STRATEGY_ID,
    ai_theme_large_cap_v20260613_signal_variant,
    build_frozen_strategy_signal,
)
from backtest_lab.regime_mode_switch import RegimeModeSwitchVariant, frozen_cycle_proven_top1_v1_variant
from backtest_lab.stock_pool_candidate_review import (
    build_candidate_review,
    load_core_defensive_candidate_source,
    write_candidate_reviews,
)
from backtest_lab.stock_pool_consensus import build_consensus, write_consensus_outputs
from backtest_lab.stock_pool_store import KNOWN_SYMBOLS, StockPoolStore
from backtest_lab.strategy_preset_dispatcher import dispatch_pool, resolve_strategy_preset
from backtest_lab.tw50_constituents import load_tw50_constituents_for_date
from backtest_lab.universal_pool_strategy import (
    PoolProfile,
    UniversalCandidateScore,
    UniversalPoolParameters,
    core_defensive_parameters_for_profile,
    default_parameters_for_profile,
    infer_pool_profile,
    score_universal_candidates,
    window_return,
)
from backtest_lab.valuation_source import ValuationSignal, load_valuation_signals


DEFAULT_OUTPUT_ROOT = "outputs/stock_pool_observations"
REPORT_NAME = "AI股票池觀察總覽"
REPORT_TITLE = "AI股票池正式觀察總覽"
REPORT_VERSION = "v20260612"
REPORT_LATEST_FILENAME = f"{REPORT_NAME}_最新版_{REPORT_VERSION}.pdf"
TARGET_STABILITY_LOW_SCORE_GAP_THRESHOLD = 0.15
FROZEN_BEST_GROUP_ID = "group_c_0050_00631l_plus_mega_caps"
TW50_ATTACK_GATE_RULE_ID = "tw50_large_breadth_attack_gate_v1"
TW50_ATTACK_GATE_BENCHMARK = "0050.TW"
TW50_ATTACK_GATE_RET60_MARGIN = 0.08
TW50_ATTACK_GATE_RET20_MIN = 0.03
TW50_ATTACK_GATE_RET60_MIN = 0.12
TW50_ATTACK_GATE_PERSISTENCE_LOOKBACK = 10
TW50_ATTACK_GATE_PERSISTENCE_MIN_DAYS = 5
CORE_DEFENSIVE_GATE_RULE_ID = "core_style_complement_opportunity_gate_v1"
CORE_DEFENSIVE_BENCHMARK = "0050.TW"
CORE_DEFENSIVE_RET60_MIN = 0.05
CORE_DEFENSIVE_RET120_MIN = 0.0
CORE_DEFENSIVE_BENCHMARK_LAG_TOLERANCE = -0.03
CORE_DEFENSIVE_RET120_BENCHMARK_LAG_TOLERANCE = -0.03
CORE_DEFENSIVE_MAX_DRAWDOWN20 = -0.12
CORE_DEFENSIVE_MAX_FLOW_RISK_SCORE = 0.35
CORE_DEFENSIVE_MARKET_EXPOSURE_BUCKET = "market_exposure_etf"
CORE_STYLE_COMPLEMENT_EXCLUDED_TICKERS = {
    "2330.TW",
    "2454.TW",
    "2308.TW",
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "6669.TW",
}
FORMAL_CANDIDATE_EXCLUDED_TICKERS = {"0050.TW"}
FORMAL_0050_EXCLUSION_POOL_IDS = {
    "large_cap_best_v20260605",
    "ai_theme_large_cap_v20260613",
    "tw50_dynamic_constituents_v0",
}
POOL_SHORT_NAMES = {
    "ai_theme_large_cap_v20260613": "AI主線池",
    "tw50_dynamic_constituents_v0": "大型廣度池",
    "large_core_bluechip_v0": "風格補強池",
}
VOTE_DIVERGENT_COLORS = ("#2457a7", "#7a3db8", "#c77917")
VOTE_WINNER_COLOR = "#13795b"
VOTE_MINOR_COLOR = "#c77917"
VOTE_NEUTRAL_COLOR = "#6b7780"
ASSET_TYPE_STOCK = "stock"
ASSET_TYPE_ETF = "etf"
ASSET_TYPE_CASH = "cash"
ASSET_TYPE_UNKNOWN = "unknown"
SELECTION_FORMAL_CANDIDATE = "formal_candidate"
SELECTION_MARKET_EXPOSURE_TOOL = "market_exposure_tool"
SELECTION_OBSERVATION_ONLY = "observation_only"
SELECTION_NO_SELECTION = "no_selection"


@dataclass(frozen=True)
class StockPoolObservation:
    schema_version: int
    pool_id: str
    pool_name: str
    strategy_preset: str
    signal_date: str
    data_end_date: str
    candidate_count: int
    passed_count: int
    pool_profile: PoolProfile
    parameters: UniversalPoolParameters
    top_ticker: str | None
    top_display: str | None
    top_score: float | None
    action_state: str
    candidates: list[UniversalCandidateScore]
    rank_score: float | None = None
    base_pool_passed: bool = False
    gate_rule_id: str = ""
    gate_reason: str = ""
    top_asset_type: str | None = None
    attack_gate_open: bool | None = None
    eligible_for_pool_selection: bool = False
    selection_layer: str = SELECTION_NO_SELECTION
    selection_reason: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    decision_layer: str = CANDIDATE_SOURCE
    active_in_trade_decision: bool = False
    source_module: str = "stock_pool_observation"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pool_profile"] = asdict(self.pool_profile)
        payload["parameters"] = asdict(self.parameters)
        payload["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return payload


def build_stock_pool_observation(
    *,
    pool: dict[str, Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    theme_by_ticker: dict[str, str] | None = None,
    conviction_by_ticker: dict[str, float] | None = None,
    market_cap_by_ticker: dict[str, float] | None = None,
    risk_signal_by_ticker: dict[str, RiskFactorSignal] | None = None,
    valuation_signal_by_ticker: dict[str, ValuationSignal] | None = None,
    require_exact_signal_date: bool = False,
) -> StockPoolObservation:
    requested_ts = pd.Timestamp(signal_date)
    signal_ts = _resolve_signal_date(prices_by_ticker, requested_ts)
    if require_exact_signal_date and signal_ts != requested_ts.normalize():
        raise ValueError(
            f"No exact common price data for signal date {requested_ts.strftime('%Y-%m-%d')}; "
            f"latest common date is {signal_ts.strftime('%Y-%m-%d')}"
        )
    raw_symbols = [
        symbol
        for symbol in pool.get("resolved_symbols") or pool.get("symbols") or []
        if symbol.get("ticker") in prices_by_ticker
    ]
    available_symbols = [
        symbol
        for symbol in raw_symbols
        if not _exclude_from_formal_candidate_universe(pool, str(symbol.get("ticker") or ""))
    ]
    candidate_prices = {
        symbol["ticker"]: prices_by_ticker[symbol["ticker"]]
        for symbol in available_symbols
    }
    profile = infer_pool_profile(candidate_prices, signal_ts, theme_by_ticker=theme_by_ticker)
    params, enforce_pool_parameters = _parameters_for_pool_strategy(pool, profile)
    scored = score_universal_candidates(
        candidate_prices,
        signal_ts,
        params,
        conviction_by_ticker=conviction_by_ticker,
        market_cap_by_ticker={
            **(market_cap_by_ticker or {}),
            **_market_cap_by_ticker(available_symbols),
        },
        risk_signal_by_ticker=risk_signal_by_ticker,
        valuation_signal_by_ticker=valuation_signal_by_ticker,
        enforce_pool_parameters=enforce_pool_parameters,
    )
    candidates = sorted(
        scored.values(),
        key=lambda item: (item.passed, item.score, item.ret20, item.ticker),
        reverse=True,
    )
    display_by_ticker = {
        symbol["ticker"]: _normalize_display_label(symbol.get("display") or symbol["ticker"], symbol["ticker"])
        for symbol in available_symbols
    }
    asset_type_by_ticker = _asset_type_by_ticker(available_symbols)
    gate_rule_id = _gate_rule_id_for_pool(pool)
    gate_details_by_ticker = _pool_gate_details_by_ticker(
        pool=pool,
        candidates=candidates,
        asset_type_by_ticker=asset_type_by_ticker,
        gate_rule_id=gate_rule_id,
        prices_by_ticker=prices_by_ticker,
        signal_date=signal_ts,
    )
    top = _first_eligible_candidate(
        candidates,
        asset_type_by_ticker=asset_type_by_ticker,
        gate_rule_id=gate_rule_id,
        gate_details_by_ticker=gate_details_by_ticker,
    )
    top_asset_type = _asset_type_for_ticker(top.ticker, asset_type_by_ticker) if top else None
    top_gate = gate_details_by_ticker.get(top.ticker) if top else None
    top_gate = top_gate or _candidate_gate_evaluation(top, top_asset_type, gate_rule_id=gate_rule_id)
    source_metadata = _build_pool_source_metadata(pool, available_symbols)
    source_metadata["candidate_asset_types"] = asset_type_by_ticker
    source_metadata["gate_rule_id"] = gate_rule_id
    source_metadata["candidate_gate_details"] = gate_details_by_ticker
    return StockPoolObservation(
        schema_version=1,
        pool_id=str(pool["pool_id"]),
        pool_name=str(pool["name"]),
        strategy_preset=str(pool.get("strategy_preset") or "universal_pool_custom"),
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        data_end_date=signal_ts.strftime("%Y-%m-%d"),
        candidate_count=len(candidates),
        passed_count=sum(1 for candidate in candidates if candidate.passed),
        pool_profile=profile,
        parameters=params,
        top_ticker=top.ticker if top else None,
        top_display=display_by_ticker.get(top.ticker, top.ticker) if top else None,
        top_score=round(top.score, 6) if top else None,
        rank_score=round(top.score, 6) if top else None,
        base_pool_passed=top_gate["base_pool_passed"],
        gate_rule_id=top_gate["gate_rule_id"],
        gate_reason=top_gate["gate_reason"],
        action_state="watch_candidate" if top else "no_valid_candidate",
        candidates=candidates,
        top_asset_type=top_asset_type,
        attack_gate_open=top_gate["attack_gate_open"],
        eligible_for_pool_selection=top_gate["eligible_for_pool_selection"],
        selection_layer=top_gate["selection_layer"],
        selection_reason=top_gate["gate_reason"] if top else "池內沒有通過入選條件的正式候選或市場曝險工具。",
        source_metadata=source_metadata,
        decision_layer=CANDIDATE_SOURCE,
        active_in_trade_decision=False,
        source_module="stock_pool_observation",
    )


def build_dispatched_stock_pool_observation(
    *,
    pool: dict[str, Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    warmup_start: str,
    theme_by_ticker: dict[str, str] | None = None,
    conviction_by_ticker: dict[str, float] | None = None,
    market_cap_by_ticker: dict[str, float] | None = None,
    risk_signal_by_ticker: dict[str, RiskFactorSignal] | None = None,
    valuation_signal_by_ticker: dict[str, ValuationSignal] | None = None,
    require_exact_signal_date: bool = False,
) -> StockPoolObservation:
    spec = resolve_strategy_preset(pool.get("strategy_preset"))
    if spec.preset in {"best_v20260605", "ai_theme_large_cap_v20260613"}:
        variant = (
            ai_theme_large_cap_v20260613_signal_variant()
            if spec.preset == "ai_theme_large_cap_v20260613"
            else frozen_cycle_proven_top1_v1_variant()
        )
        return _build_regime_signal_observation(
            pool=pool,
            prices_by_ticker=prices_by_ticker,
            signal_date=signal_date,
            warmup_start=warmup_start,
            market_cap_by_ticker=market_cap_by_ticker,
            risk_signal_by_ticker=risk_signal_by_ticker,
            valuation_signal_by_ticker=valuation_signal_by_ticker,
            require_exact_signal_date=require_exact_signal_date,
            variant=variant,
            strategy_id=AI_THEME_STRATEGY_ID if spec.preset == "ai_theme_large_cap_v20260613" else STRATEGY_ID,
        )
    return build_stock_pool_observation(
        pool=pool,
        prices_by_ticker=prices_by_ticker,
        signal_date=signal_date,
        theme_by_ticker=theme_by_ticker,
        conviction_by_ticker=conviction_by_ticker,
        market_cap_by_ticker=market_cap_by_ticker,
        risk_signal_by_ticker=risk_signal_by_ticker,
        valuation_signal_by_ticker=valuation_signal_by_ticker,
        require_exact_signal_date=require_exact_signal_date,
    )


def _build_pool_source_metadata(pool: dict[str, Any], symbols: list[dict[str, Any]]) -> dict[str, Any]:
    if pool.get("strategy_preset") == "core_defensive_style_v1":
        return {
            "source_type": "core_defensive_style_representatives",
            "source_path": pool.get("core_defensive_style_source", pool.get("candidate_review_config", {}).get("path", "")),
            "selection_mode": pool.get(
                "core_defensive_style_selection_mode",
                "one_representative_per_style_bucket_v1",
            ),
            "source_candidate_count": pool.get("core_defensive_source_candidate_count", len(symbols)),
            "representative_count": len(symbols),
            "style_buckets": pool.get("core_defensive_style_buckets", []),
            "candidate_displays": [symbol.get("display") or symbol.get("ticker") for symbol in symbols],
        }
    if pool.get("dynamic_constituents", {}).get("source") == "tw50_history_csv":
        return {
            "source_type": "tw50_constituents",
            "source_path": pool.get("tw50_constituent_source", pool.get("dynamic_constituents", {}).get("path", "")),
            "candidate_count": len(symbols),
            "candidate_displays": [symbol.get("display") or symbol.get("ticker") for symbol in symbols],
        }
    if pool.get("strategy_preset") != "radar_core_mid_small_calibrated_v1":
        return {}
    formal_dates = sorted(
        {
            str(symbol.get("formal_report_date") or "").strip()
            for symbol in symbols
            if str(symbol.get("formal_report_date") or "").strip()
        }
    )
    return {
        "source_type": "formal_radar_candidates",
        "source_path": pool.get("radar_candidate_source", ""),
        "mode": pool.get("radar_candidate_mode", ""),
        "formal_report_dates": formal_dates,
        "candidate_count": len(symbols),
        "candidate_displays": [symbol.get("display") or symbol.get("ticker") for symbol in symbols],
        "candidate_symbols": [symbol.get("symbol") or str(symbol.get("ticker", "")).split(".")[0] for symbol in symbols],
    }


def _exclude_from_formal_candidate_universe(pool: dict[str, Any], ticker: str) -> bool:
    normalized = _normalize_ticker(ticker)
    if normalized not in FORMAL_CANDIDATE_EXCLUDED_TICKERS:
        return False
    pool_id = str(pool.get("pool_id") or "")
    preset = str(pool.get("strategy_preset") or "")
    dynamic_source = str((pool.get("dynamic_constituents") or {}).get("source") or "")
    return (
        pool_id in FORMAL_0050_EXCLUSION_POOL_IDS
        or preset in {"best_v20260605", "ai_theme_large_cap_v20260613"}
        or dynamic_source == "tw50_history_csv"
    )


def _normalize_ticker(ticker: str) -> str:
    text = str(ticker or "").strip()
    if not text:
        return text
    if "." not in text and text.upper() != "CASH":
        return f"{text}.TW"
    return text


def _market_cap_by_ticker(symbols: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol in symbols:
        ticker = str(symbol.get("ticker") or "").strip()
        if not ticker:
            continue
        market_cap = _number(symbol.get("free_float_market_cap_twd") or symbol.get("market_cap_twd"))
        if market_cap > 0:
            result[ticker] = market_cap
    return result


def _asset_type_by_ticker(symbols: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for symbol in symbols:
        ticker = str(symbol.get("ticker") or "").strip()
        if not ticker:
            continue
        result[ticker] = _normalize_asset_type(symbol.get("asset_type"), ticker)
    return result


def _normalize_asset_type(value: object, ticker: str = "") -> str:
    text = str(value or "").strip().lower()
    if text in {ASSET_TYPE_STOCK, ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
        return text
    if ticker:
        known = KNOWN_SYMBOLS.get(ticker, {})
        known_type = str(known.get("asset_type") or "").strip().lower()
        if known_type in {ASSET_TYPE_STOCK, ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
            return known_type
    if ticker.lower() == "cash":
        return ASSET_TYPE_CASH
    return ASSET_TYPE_UNKNOWN


def _asset_type_for_ticker(ticker: str | None, asset_type_by_ticker: dict[str, str] | None = None) -> str:
    if not ticker:
        return ASSET_TYPE_UNKNOWN
    asset_type_by_ticker = asset_type_by_ticker or {}
    return asset_type_by_ticker.get(ticker) or _normalize_asset_type(None, ticker)


def _first_eligible_candidate(
    candidates: list[UniversalCandidateScore],
    *,
    asset_type_by_ticker: dict[str, str],
    gate_rule_id: str,
    gate_details_by_ticker: dict[str, dict[str, Any]] | None = None,
) -> UniversalCandidateScore | None:
    gate_details_by_ticker = gate_details_by_ticker or {}
    if gate_rule_id == TW50_ATTACK_GATE_RULE_ID:
        top = candidates[0] if candidates else None
        if top and (gate_details_by_ticker.get(top.ticker) or {}).get("eligible_for_pool_selection"):
            return top
        return None
    if gate_rule_id == CORE_DEFENSIVE_GATE_RULE_ID:
        stock_candidate = next(
            (
                candidate
                for candidate in candidates
                if _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker) == ASSET_TYPE_STOCK
                and (gate_details_by_ticker.get(candidate.ticker) or {}).get("eligible_for_pool_selection")
            ),
            None,
        )
        if stock_candidate:
            return stock_candidate
        return next(
            (
                candidate
                for candidate in candidates
                if (gate_details_by_ticker.get(candidate.ticker) or {}).get("selection_layer")
                == SELECTION_MARKET_EXPOSURE_TOOL
                and (gate_details_by_ticker.get(candidate.ticker) or {}).get("eligible_for_pool_selection")
            ),
            None,
        )
    return next(
        (
            candidate
            for candidate in candidates
            if (gate_details_by_ticker.get(candidate.ticker) or _candidate_gate_evaluation(
                    candidate,
                    _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker),
                    gate_rule_id=gate_rule_id,
                ))["eligible_for_pool_selection"]
        ),
        None,
    )


def _gate_rule_id_for_pool(pool: dict[str, Any]) -> str:
    pool_id = str(pool.get("pool_id") or "")
    preset = str(pool.get("strategy_preset") or "universal_pool_custom")
    if pool_id == "tw50_dynamic_constituents_v0":
        return TW50_ATTACK_GATE_RULE_ID
    if preset == "core_defensive_style_v1":
        return CORE_DEFENSIVE_GATE_RULE_ID
    if preset in {"best_v20260605", "ai_theme_large_cap_v20260613"}:
        return f"{preset}_formal_regime_gate"
    return "universal_pool_base_gate_v1"


def _pool_gate_details_by_ticker(
    *,
    pool: dict[str, Any],
    candidates: list[UniversalCandidateScore],
    asset_type_by_ticker: dict[str, str],
    gate_rule_id: str,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    if gate_rule_id != TW50_ATTACK_GATE_RULE_ID:
        if gate_rule_id == CORE_DEFENSIVE_GATE_RULE_ID:
            for candidate in candidates:
                asset_type = _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker)
                details[candidate.ticker] = _core_defensive_gate_evaluation(
                    candidate=candidate,
                    asset_type=asset_type,
                    prices_by_ticker=prices_by_ticker,
                    signal_date=signal_date,
                )
            return details
        for candidate in candidates:
            asset_type = _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker)
            details[candidate.ticker] = _candidate_gate_evaluation(candidate, asset_type, gate_rule_id=gate_rule_id)
        return details

    for candidate in candidates:
        asset_type = _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker)
        details[candidate.ticker] = _tw50_attack_gate_evaluation(
            candidate=candidate,
            asset_type=asset_type,
            candidates=candidates,
            prices_by_ticker=prices_by_ticker,
            signal_date=signal_date,
        )
    return details


def _tw50_attack_gate_evaluation(
    *,
    candidate: UniversalCandidateScore,
    asset_type: str | None,
    candidates: list[UniversalCandidateScore],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> dict[str, Any]:
    base_pool_passed = bool(candidate.passed)
    normalized_type = _normalize_asset_type(asset_type)
    if normalized_type in {ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
        return _candidate_gate_evaluation(candidate, asset_type, gate_rule_id=TW50_ATTACK_GATE_RULE_ID)

    benchmark_ret60 = _window_return_on_or_before(
        prices_by_ticker.get(TW50_ATTACK_GATE_BENCHMARK),
        signal_date,
        60,
    )
    if benchmark_ret60 is None:
        return _tw50_gate_result(
            candidate,
            attack_gate_open=False,
            eligible=False,
            benchmark_margin_passed=False,
            momentum_quality_passed=False,
            persistence_passed=False,
            gate_reason="大型廣度池 v1 未通過：缺少 0050 benchmark 價格，不能確認相對超額。",
        )

    ret60_margin = candidate.ret60 - benchmark_ret60
    benchmark_margin_passed = ret60_margin >= TW50_ATTACK_GATE_RET60_MARGIN
    momentum_quality_passed = (
        candidate.ret20 >= TW50_ATTACK_GATE_RET20_MIN
        and candidate.ret60 >= TW50_ATTACK_GATE_RET60_MIN
        and candidate.ret60 > 0
    )
    persistence_days, persistence_total = _tw50_persistence_days(
        ticker=candidate.ticker,
        candidates=candidates,
        prices_by_ticker=prices_by_ticker,
        signal_date=signal_date,
    )
    persistence_passed = persistence_days >= TW50_ATTACK_GATE_PERSISTENCE_MIN_DAYS
    attack_gate_open = bool(
        base_pool_passed
        and benchmark_margin_passed
        and momentum_quality_passed
        and persistence_passed
    )
    reason_parts = [
        f"base={'Y' if base_pool_passed else 'N'}",
        f"60日相對0050超額={ret60_margin:.1%}({'Y' if benchmark_margin_passed else 'N'})",
        f"20/60動能品質={'Y' if momentum_quality_passed else 'N'}",
        f"持續性={persistence_days}/{persistence_total}日({'Y' if persistence_passed else 'N'})",
    ]
    return _tw50_gate_result(
        candidate,
        attack_gate_open=attack_gate_open,
        eligible=attack_gate_open,
        benchmark_margin_passed=benchmark_margin_passed,
        momentum_quality_passed=momentum_quality_passed,
        persistence_passed=persistence_passed,
        gate_reason="大型廣度池 v1：" + "；".join(reason_parts),
    )


def _tw50_gate_result(
    candidate: UniversalCandidateScore,
    *,
    attack_gate_open: bool,
    eligible: bool,
    benchmark_margin_passed: bool,
    momentum_quality_passed: bool,
    persistence_passed: bool,
    gate_reason: str,
) -> dict[str, Any]:
    return {
        "rank_score": round(candidate.score, 6),
        "base_pool_passed": bool(candidate.passed),
        "benchmark_margin_passed": benchmark_margin_passed,
        "momentum_quality_passed": momentum_quality_passed,
        "persistence_passed": persistence_passed,
        "attack_gate_open": attack_gate_open,
        "eligible_for_pool_selection": eligible,
        "selection_layer": SELECTION_FORMAL_CANDIDATE if eligible else SELECTION_OBSERVATION_ONLY,
        "gate_rule_id": TW50_ATTACK_GATE_RULE_ID,
        "gate_reason": gate_reason,
    }


def _core_defensive_gate_evaluation(
    *,
    candidate: UniversalCandidateScore,
    asset_type: str | None,
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> dict[str, Any]:
    normalized_type = _normalize_asset_type(asset_type)
    if normalized_type in {ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
        return _core_defensive_market_exposure_gate_result(candidate)

    benchmark_ret60 = _window_return_on_or_before(
        prices_by_ticker.get(CORE_DEFENSIVE_BENCHMARK),
        signal_date,
        60,
    )
    if benchmark_ret60 is None:
        return _core_defensive_gate_result(
            candidate,
            attack_gate_open=False,
            eligible=False,
            benchmark_resilience_passed=False,
            trend_resilience_passed=False,
            opportunity_cost_passed=False,
            drawdown_control_passed=False,
            risk_control_passed=False,
            gate_reason="風格補強池 v1 未通過：缺少 0050 benchmark 價格，不能確認相對強度。",
        )

    benchmark_ret120 = _window_return_on_or_before(
        prices_by_ticker.get(CORE_DEFENSIVE_BENCHMARK),
        signal_date,
        120,
    )
    if benchmark_ret120 is None:
        return _core_defensive_gate_result(
            candidate,
            attack_gate_open=False,
            eligible=False,
            benchmark_resilience_passed=False,
            trend_resilience_passed=False,
            opportunity_cost_passed=False,
            drawdown_control_passed=False,
            risk_control_passed=False,
            gate_reason="風格補強池 v1 未通過：缺少 0050 120日 benchmark，不能確認機會成本。",
        )

    benchmark_resilience_passed = candidate.ret60 - benchmark_ret60 >= CORE_DEFENSIVE_BENCHMARK_LAG_TOLERANCE
    trend_resilience_passed = (
        candidate.ret60 >= CORE_DEFENSIVE_RET60_MIN
        and candidate.ret120 >= CORE_DEFENSIVE_RET120_MIN
    )
    opportunity_cost_passed = candidate.ret120 - benchmark_ret120 >= CORE_DEFENSIVE_RET120_BENCHMARK_LAG_TOLERANCE
    drawdown_control_passed = candidate.drawdown20 >= CORE_DEFENSIVE_MAX_DRAWDOWN20
    risk_control_passed = candidate.flow_risk_score <= CORE_DEFENSIVE_MAX_FLOW_RISK_SCORE
    attack_gate_open = bool(
        candidate.passed
        and benchmark_resilience_passed
        and trend_resilience_passed
        and opportunity_cost_passed
        and drawdown_control_passed
        and risk_control_passed
    )
    reason_parts = [
        f"base={'Y' if candidate.passed else 'N'}",
        f"60日相對0050強度={candidate.ret60 - benchmark_ret60:.1%}({'Y' if benchmark_resilience_passed else 'N'})",
        f"60/120中期上攻力={'Y' if trend_resilience_passed else 'N'}",
        f"120日機會成本={candidate.ret120 - benchmark_ret120:.1%}({'Y' if opportunity_cost_passed else 'N'})",
        f"20日回撤控管={candidate.drawdown20:.1%}({'Y' if drawdown_control_passed else 'N'})",
        f"籌碼風險={candidate.flow_risk_score:.2f}({'Y' if risk_control_passed else 'N'})",
    ]
    return _core_defensive_gate_result(
        candidate,
        attack_gate_open=attack_gate_open,
        eligible=attack_gate_open,
        benchmark_resilience_passed=benchmark_resilience_passed,
        trend_resilience_passed=trend_resilience_passed,
        opportunity_cost_passed=opportunity_cost_passed,
        drawdown_control_passed=drawdown_control_passed,
        risk_control_passed=risk_control_passed,
        gate_reason="風格補強池 v1：" + "；".join(reason_parts),
    )


def _core_defensive_gate_result(
    candidate: UniversalCandidateScore,
    *,
    attack_gate_open: bool,
    eligible: bool,
    benchmark_resilience_passed: bool,
    trend_resilience_passed: bool,
    opportunity_cost_passed: bool,
    drawdown_control_passed: bool,
    risk_control_passed: bool,
    gate_reason: str,
) -> dict[str, Any]:
    return {
        "rank_score": round(candidate.score, 6),
        "base_pool_passed": bool(candidate.passed),
        "benchmark_resilience_passed": benchmark_resilience_passed,
        "trend_resilience_passed": trend_resilience_passed,
        "opportunity_cost_passed": opportunity_cost_passed,
        "drawdown_control_passed": drawdown_control_passed,
        "risk_control_passed": risk_control_passed,
        "attack_gate_open": attack_gate_open,
        "eligible_for_pool_selection": eligible,
        "selection_layer": SELECTION_FORMAL_CANDIDATE if eligible else SELECTION_OBSERVATION_ONLY,
        "gate_rule_id": CORE_DEFENSIVE_GATE_RULE_ID,
        "gate_reason": gate_reason,
    }


def _core_defensive_market_exposure_gate_result(candidate: UniversalCandidateScore) -> dict[str, Any]:
    eligible = bool(candidate.passed)
    return {
        "rank_score": round(candidate.score, 6),
        "base_pool_passed": bool(candidate.passed),
        "benchmark_resilience_passed": None,
        "trend_resilience_passed": None,
        "opportunity_cost_passed": None,
        "drawdown_control_passed": None,
        "risk_control_passed": None,
        "attack_gate_open": None,
        "eligible_for_pool_selection": eligible,
        "selection_layer": SELECTION_MARKET_EXPOSURE_TOOL if eligible else SELECTION_OBSERVATION_ONLY,
        "gate_rule_id": CORE_DEFENSIVE_GATE_RULE_ID,
        "gate_reason": (
            "風格補強池 v1：非AI風格個股無合格時，通過池內條件的 ETF 可作市場曝險替代；"
            f"base={'Y' if candidate.passed else 'N'}。"
        ),
    }


def _tw50_persistence_days(
    *,
    ticker: str,
    candidates: list[UniversalCandidateScore],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> tuple[int, int]:
    benchmark = prices_by_ticker.get(TW50_ATTACK_GATE_BENCHMARK)
    if benchmark is None or ticker not in prices_by_ticker:
        return 0, 0
    candidate_tickers = [candidate.ticker for candidate in candidates if candidate.ticker in prices_by_ticker]
    trading_dates = sorted(
        set.intersection(
            *(set(prices_by_ticker[item].index[prices_by_ticker[item].index <= signal_date]) for item in candidate_tickers),
            set(benchmark.index[benchmark.index <= signal_date]),
        )
    )
    if not trading_dates:
        return 0, 0
    dates = trading_dates[-TW50_ATTACK_GATE_PERSISTENCE_LOOKBACK:]
    pass_days = 0
    evaluated = 0
    for current_date in dates:
        benchmark_ret60 = _window_return_on_or_before(benchmark, current_date, 60)
        if benchmark_ret60 is None:
            continue
        rows = []
        for item in candidate_tickers:
            ret60 = _window_return_on_or_before(prices_by_ticker[item], current_date, 60)
            if ret60 is None:
                continue
            rows.append((item, ret60))
        if not rows:
            continue
        evaluated += 1
        rows.sort(key=lambda row: row[1], reverse=True)
        top_cutoff = max(5, int(len(rows) * 0.2 + 0.9999))
        top_tickers = {item for item, _ in rows[:top_cutoff]}
        ticker_ret60 = dict(rows).get(ticker)
        if (
            ticker in top_tickers
            and ticker_ret60 is not None
            and ticker_ret60 - benchmark_ret60 >= TW50_ATTACK_GATE_RET60_MARGIN
        ):
            pass_days += 1
    return pass_days, evaluated


def _window_return_on_or_before(frame: pd.DataFrame | None, signal_date: pd.Timestamp, window: int) -> float | None:
    if frame is None or frame.empty:
        return None
    history = frame.loc[frame.index <= signal_date].dropna(subset=["adj_close"])
    if len(history) <= window:
        return None
    return window_return(history["adj_close"], window)


def _candidate_gate_evaluation(
    candidate: UniversalCandidateScore | None,
    asset_type: str | None,
    *,
    gate_rule_id: str,
    attack_gate_active: bool | None = None,
) -> dict[str, Any]:
    base_pool_passed = bool(candidate and candidate.passed)
    attack_gate_open = _attack_gate_open_for_candidate(
        candidate,
        asset_type,
        attack_gate_active=attack_gate_active,
    )
    selection_layer = _selection_layer_for_candidate(candidate, asset_type)
    eligible = selection_layer in {SELECTION_FORMAL_CANDIDATE, SELECTION_MARKET_EXPOSURE_TOOL}
    if candidate is None:
        gate_reason = "池內沒有候選資料。"
    elif gate_rule_id == TW50_ATTACK_GATE_RULE_ID:
        gate_reason = (
            "大型廣度池 v1：需另行檢查 benchmark margin、20/60 動能品質與持續性。"
            if base_pool_passed
            else f"大型廣度池 v1 未通過：{candidate.reason or '池內基本條件未通過'}。"
        )
        attack_gate_open = False
        eligible = False
        selection_layer = SELECTION_OBSERVATION_ONLY
    elif gate_rule_id == CORE_DEFENSIVE_GATE_RULE_ID:
        gate_reason = (
            "風格補強池 v1：需另行檢查 benchmark 相對強度、60/120 中期上攻力、回撤與籌碼風險。"
            if base_pool_passed
            else f"風格補強池 v1 未通過：{candidate.reason or '池內基本條件未通過'}。"
        )
        attack_gate_open = False
        eligible = False
        selection_layer = SELECTION_OBSERVATION_ONLY
    elif gate_rule_id.endswith("_formal_regime_gate"):
        if _normalize_asset_type(asset_type) in {ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
            gate_reason = "正式引擎目前選擇市場曝險工具；ETF 不套用個股攻擊閘門。"
        elif attack_gate_open:
            gate_reason = "正式引擎個股攻擊閘門已開啟，且該個股為模型正式目標。"
        else:
            gate_reason = f"正式引擎個股攻擊閘門未開啟或非正式目標：{candidate.reason or '觀察'}。"
    else:
        gate_reason = (
            "通用池基礎 gate：通過池內硬條件。"
            if base_pool_passed
            else f"通用池基礎 gate 未通過：{candidate.reason or '池內基本條件未通過'}。"
        )
    return {
        "rank_score": round(candidate.score, 6) if candidate else None,
        "base_pool_passed": base_pool_passed,
        "attack_gate_open": attack_gate_open,
        "eligible_for_pool_selection": eligible,
        "selection_layer": selection_layer,
        "gate_rule_id": gate_rule_id,
        "gate_reason": gate_reason,
    }


def _eligible_for_pool_selection(
    candidate: UniversalCandidateScore | None,
    asset_type: str | None,
    *,
    gate_rule_id: str = "universal_pool_base_gate_v1",
) -> bool:
    if candidate is None:
        return False
    return bool(
        _candidate_gate_evaluation(
            candidate,
            asset_type,
            gate_rule_id=gate_rule_id,
        )["eligible_for_pool_selection"]
    )


def _selection_layer_for_candidate(candidate: UniversalCandidateScore | None, asset_type: str | None) -> str:
    if candidate is None:
        return SELECTION_NO_SELECTION
    normalized_type = _normalize_asset_type(asset_type)
    if normalized_type in {ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
        return SELECTION_MARKET_EXPOSURE_TOOL if candidate.passed else SELECTION_OBSERVATION_ONLY
    return SELECTION_FORMAL_CANDIDATE if candidate.passed else SELECTION_OBSERVATION_ONLY


def _attack_gate_open_for_candidate(
    candidate: UniversalCandidateScore | None,
    asset_type: str | None,
    *,
    attack_gate_active: bool | None = None,
) -> bool | None:
    if candidate is None:
        return None
    normalized_type = _normalize_asset_type(asset_type)
    if normalized_type in {ASSET_TYPE_ETF, ASSET_TYPE_CASH}:
        return None
    if attack_gate_active is not None:
        return bool(attack_gate_active and candidate.passed)
    return bool(candidate.passed)


def _selection_reason_for_candidate(candidate: UniversalCandidateScore | None, asset_type: str | None) -> str:
    if candidate is None:
        return "池內沒有通過入選條件的正式候選或市場曝險工具。"
    layer = _selection_layer_for_candidate(candidate, asset_type)
    if layer == SELECTION_MARKET_EXPOSURE_TOOL:
        return "市場曝險工具依池內市場狀態或策略規則入選，不套用個股攻擊閘門。"
    if layer == SELECTION_FORMAL_CANDIDATE:
        return "個股已通過池內攻擊條件，可作為該池正式候選。"
    return "僅為觀察排名；尚未通過池內攻擊條件，不列入該池投票。"


def _parameters_for_pool_strategy(pool: dict[str, Any], profile: PoolProfile) -> tuple[UniversalPoolParameters, bool]:
    preset = str(pool.get("strategy_preset") or "universal_pool_custom")
    if preset == "core_defensive_style_v1":
        return core_defensive_parameters_for_profile(profile), True
    return default_parameters_for_profile(profile), False


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def _source_summary(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    if metadata.get("source_type") == "formal_radar_candidates":
        date_text = ",".join(metadata.get("formal_report_dates") or []) or "未標日期"
        candidates = "、".join(metadata.get("candidate_displays") or [])
        return f"RADAR正式候選 {date_text}：{candidates}"
    if metadata.get("source_type") == "tw50_constituents":
        return f"大型權值成分股：{metadata.get('candidate_count', 0)}檔，來源 {metadata.get('source_path') or '未標'}"
    if metadata.get("source_type") == "core_defensive_style_representatives":
        return (
            f"風格補強代表：{metadata.get('representative_count', 0)}檔，"
            f"模式 {metadata.get('selection_mode') or '未標'}"
        )
    if metadata.get("source_type") in {"best_v20260605_signal", "ai_theme_large_cap_v20260613_signal"}:
        gate_text = "個股攻擊閘門已開啟" if metadata.get("attack_gate_active") else "個股攻擊閘門未開啟"
        return f"最佳版正式引擎：{metadata.get('market_regime_label') or '市場環境未標'}，{gate_text}"
    return str(metadata.get("source_type") or "")


def write_stock_pool_observation(output_dir: Path, observation: StockPoolObservation) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = observation.to_dict()
    (output_dir / "stock_pool_observation.json").write_text(
        json.dumps({"status": "ready", "observation": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(payload["candidates"]).to_csv(
        output_dir / "stock_pool_observation_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_stock_pool_observation_batch(
    *,
    pools: list[dict[str, Any]],
    signal_date: str,
    warmup_start: str,
    cache_dir: str | Path,
    output_root: str | Path,
    radar_snapshot_dir: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
    market_cap_data: str | Path | None = None,
    institutional_flow_data: str | Path | None = None,
    margin_short_data: str | Path | None = None,
    borrow_lending_data: str | Path | None = None,
    day_trading_data: str | Path | None = None,
    sentiment_data: str | Path | None = None,
    valuation_data: str | Path | None = None,
    tw50_constituents_path: str | Path | None = None,
    radar_top_n: int = 20,
    require_exact_signal_date: bool = False,
    require_fresh_institutional_flow: bool = False,
    operational_only: bool = True,
) -> dict[str, Any]:
    date_key = pd.Timestamp(signal_date).strftime("%Y%m%d")
    root = Path(output_root) / date_key
    root.mkdir(parents=True, exist_ok=True)
    market_caps, market_cap_source = load_first_available_market_caps(
        signal_date=signal_date,
        explicit_path=market_cap_data,
        radar_data_dir=radar_data_dir,
    )
    risk_signals, risk_sources = load_first_available_risk_factors(
        signal_date=signal_date,
        radar_data_dir=radar_data_dir,
        institutional_path=institutional_flow_data,
        margin_short_path=margin_short_data,
        borrow_lending_path=borrow_lending_data,
        day_trading_path=day_trading_data,
        sentiment_path=sentiment_data,
    )
    manifest: dict[str, Any] = {
        "status": "ready",
        "requested_signal_date": signal_date,
        "signal_date": signal_date,
        "actual_signal_date": signal_date,
        "signal_date_fallback_used": False,
        "fallback_reason": "",
        "require_exact_signal_date": require_exact_signal_date,
        "require_fresh_institutional_flow": require_fresh_institutional_flow,
        "operational_only": operational_only,
        "market_cap_source": market_cap_source,
        "market_cap_count": len(market_caps),
        "risk_factor_sources": risk_sources,
        "risk_factor_count": len(risk_signals),
        "risk_factor_coverage_end_by_kind": _risk_factor_coverage_end_by_kind(risk_signals),
        "valuation_source": str(valuation_data or ""),
        "output_root": str(root),
        "generated": [],
        "skipped": [],
    }
    for pool in _observation_pools(pools, operational_only=operational_only):
        pool = _resolve_dynamic_observation_pool(
            pool,
            signal_date=signal_date,
            radar_snapshot_dir=radar_snapshot_dir,
            radar_data_dir=radar_data_dir,
            radar_top_n=radar_top_n,
            tw50_constituents_path=tw50_constituents_path,
        )
        tickers = [symbol["ticker"] for symbol in pool.get("resolved_symbols", [])]
        if not tickers:
            reason = (
                "missing_formal_radar_candidates"
                if pool.get("strategy_preset") == "radar_core_mid_small_calibrated_v1"
                else "no_resolved_symbols"
            )
            manifest["skipped"].append(
                {
                    "pool_id": pool.get("pool_id"),
                    "pool_name": pool.get("name"),
                    "role_name": pool.get("role_name", ""),
                    "role_description": pool.get("role_description", ""),
                    "candidate_review_frequency": pool.get("candidate_review_frequency", ""),
                    "candidate_update_policy": pool.get("candidate_update_policy", ""),
                    "candidate_review": build_candidate_review(
                        pool,
                        signal_date=signal_date,
                        resolved_symbols=pool.get("resolved_symbols", []),
                    ),
                    "dispatch": dispatch_pool(pool),
                    "reason": reason,
                    "decision_layer": DATA_READINESS,
                    "active_in_trade_decision": False,
                    "source_module": "stock_pool_observation",
                    "signal_date": signal_date,
                }
            )
            continue
        try:
            price_tickers = _observation_price_tickers(pool, tickers)
            prices, missing_price_tickers = _load_observation_price_frames(
                tickers=price_tickers,
                start_date=_price_start_for_pool(pool, warmup_start),
                end_date=signal_date,
                cache_dir=cache_dir,
            )
            if not prices:
                raise ValueError(f"No price data available for pool tickers: {', '.join(tickers)}")
            valuation_signals = load_valuation_signals(
                valuation_data,
                signal_date=signal_date,
                current_price_by_ticker=_current_close_by_ticker(prices, signal_date),
            )
            observation = build_dispatched_stock_pool_observation(
                pool=pool,
                prices_by_ticker=prices,
                signal_date=signal_date,
                warmup_start=warmup_start,
                market_cap_by_ticker=market_caps,
                risk_signal_by_ticker=risk_signals,
                valuation_signal_by_ticker=valuation_signals,
                require_exact_signal_date=require_exact_signal_date,
            )
            pool_dir = root / str(pool["pool_id"])
            write_stock_pool_observation(pool_dir, observation)
            manifest["generated"].append(
                {
                    "pool_id": observation.pool_id,
                    "pool_name": observation.pool_name,
                    "role_name": pool.get("role_name", ""),
                    "role_description": pool.get("role_description", ""),
                    "candidate_review_frequency": pool.get("candidate_review_frequency", ""),
                    "candidate_update_policy": pool.get("candidate_update_policy", ""),
                    "candidate_review": build_candidate_review(
                        pool,
                        signal_date=observation.signal_date,
                        resolved_symbols=pool.get("resolved_symbols", []),
                    ),
                    "dispatch": dispatch_pool(pool),
                    "signal_date": observation.signal_date,
                    "top_ticker": observation.top_ticker,
                    "top_display": observation.top_display,
                    "top_asset_type": observation.top_asset_type,
                    "score": observation.top_score,
                    "rank_score": observation.rank_score,
                    "rank": 1 if observation.top_ticker else None,
                    "base_pool_passed": observation.base_pool_passed,
                    "attack_gate_open": observation.attack_gate_open,
                    "eligible_for_pool_selection": observation.eligible_for_pool_selection,
                    "selection_layer": observation.selection_layer,
                    "selection_reason": observation.selection_reason,
                    "gate_rule_id": observation.gate_rule_id,
                    "gate_reason": observation.gate_reason,
                    "action_state": observation.action_state,
                    "decision_layer": observation.decision_layer,
                    "active_in_trade_decision": observation.active_in_trade_decision,
                    "source_module": observation.source_module,
                    "top_candidates": _top_candidate_rows(observation),
                    "source_metadata": observation.source_metadata,
                    "vote_group": pool.get("vote_group", ""),
                    "missing_price_tickers": missing_price_tickers,
                    "output_dir": str(pool_dir),
                }
            )
        except Exception as error:  # pragma: no cover - defensive batch manifest path
            manifest["skipped"].append(
                {
                    "pool_id": pool.get("pool_id"),
                    "pool_name": pool.get("name"),
                    "role_name": pool.get("role_name", ""),
                    "role_description": pool.get("role_description", ""),
                    "candidate_review_frequency": pool.get("candidate_review_frequency", ""),
                    "candidate_update_policy": pool.get("candidate_update_policy", ""),
                    "candidate_review": build_candidate_review(
                        pool,
                        signal_date=signal_date,
                        resolved_symbols=pool.get("resolved_symbols", []),
                    ),
                    "dispatch": dispatch_pool(pool),
                    "reason": str(error),
                    "decision_layer": DATA_READINESS,
                    "active_in_trade_decision": False,
                    "source_module": "stock_pool_observation",
                    "signal_date": signal_date,
                }
            )
    _finalize_batch_signal_date_metadata(manifest)
    manifest["consensus"] = build_consensus(manifest)
    manifest["report_wording_boundary"] = _report_wording_boundary()
    _attach_cashflow_report_boundary(manifest)
    _attach_target_stability_warning_boundary(manifest)
    _attach_live_risk_regime_warning_boundary(manifest)
    _attach_chip_context_report_boundary(manifest)
    _set_formal_report_readiness(manifest)
    manifest["decision_first_report_contract"] = _decision_first_report_contract(manifest, output_root=Path(output_root))
    manifest["model_layer_audit"] = default_stock_pool_model_layer_audit(
        signal_date=manifest.get("actual_signal_date") or signal_date,
        generated_pools=manifest["generated"],
        risk_factor_sources=risk_sources,
        valuation_source=str(valuation_data or ""),
    )
    (root / "stock_pool_observation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_model_layer_audit(root / "model_layer_audit.json", manifest["model_layer_audit"])
    write_stock_pool_observation_batch_summary(root, manifest)
    write_consensus_outputs(root, manifest)
    write_candidate_reviews(root, manifest)
    return manifest


def _finalize_batch_signal_date_metadata(manifest: dict[str, Any]) -> None:
    requested = str(manifest.get("requested_signal_date") or manifest.get("signal_date") or "")
    generated_dates = sorted(
        {
            str(item.get("signal_date") or "")
            for item in manifest.get("generated", [])
            if item.get("signal_date")
        }
    )
    if not generated_dates:
        manifest["actual_signal_date"] = requested
        manifest["signal_date"] = requested
        manifest["signal_date_fallback_used"] = False
        manifest["fallback_reason"] = ""
        return
    if len(generated_dates) == 1:
        actual = generated_dates[0]
    else:
        actual = "mixed:" + ",".join(generated_dates)
    manifest["actual_signal_date"] = actual
    manifest["signal_date"] = actual
    fallback_used = actual != requested
    manifest["signal_date_fallback_used"] = fallback_used
    manifest["fallback_reason"] = (
        f"Requested signal date {requested} has no exact complete common price data; "
        f"used latest complete signal date {actual}."
        if fallback_used
        else ""
    )


def _set_formal_report_readiness(manifest: dict[str, Any]) -> None:
    visible_generated = [
        item for item in manifest.get("generated", []) if not _hide_from_formal_report(item)
    ]
    visible_skipped = [
        item for item in manifest.get("skipped", []) if not _hide_from_formal_report(item)
    ]
    blockers = [
        {
            "pool_id": item.get("pool_id", ""),
            "pool_name": item.get("pool_name") or item.get("name") or "",
            "reason": item.get("reason", ""),
            "reason_zh": _sanitize_visible_report_reason(item.get("reason", "")),
        }
        for item in visible_skipped
    ]
    for item in visible_generated:
        missing = _visible_missing_price_tickers(item.get("missing_price_tickers") or [])
        if not missing:
            continue
        blockers.append(
            {
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name") or item.get("name") or "",
                "reason": "missing_price_tickers:" + ",".join(missing),
                "reason_zh": "資料不足：部分正式候選缺少價格資料（" + "、".join(missing) + "）。",
            }
        )
    chip_context = manifest.get("chip_context") or {}
    if (
        manifest.get("require_fresh_institutional_flow")
        and visible_generated
        and chip_context.get("chip_context_state") in {"chip_data_insufficient", "chip_not_available"}
    ):
        blockers.append(
            {
                "pool_id": "chip_context",
                "pool_name": "籌碼資料",
                "reason": str(chip_context.get("chip_context_reason") or "chip_context_data_insufficient"),
                "reason_zh": str(chip_context.get("chip_context_reason") or "資料不足：籌碼資料尚未補齊。"),
            }
        )
    ready = bool(visible_generated) and not blockers
    manifest["formal_report_ready"] = ready
    manifest["formal_report_blocker_count"] = len(blockers)
    manifest["formal_report_blockers"] = blockers
    manifest["formal_report_blocked_reason"] = (
        ""
        if ready
        else "正式可見觀察池資料不足，停止產生最新版 PDF，避免用不完整資料覆蓋雲端報告。"
    )


def _report_wording_boundary() -> dict[str, Any]:
    return {
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "report_boundary": "pool1_pool2_formal_baseline_with_report_only_diagnostics",
        "formal_baseline": {
            "label": "正式模型基準",
            "description": formal_model_report_description(),
            "active_in_trade_decision": True,
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "components": ["pool1_primary_selector", "pool2_tw50_pit_ready_confirmation_risk_layer", "combined_cap40_confirmation1_base"],
        },
        "diagnostic_boundary": {
            "label": "診斷註解",
            "description": "非正式診斷註解目前只作內部檢查或風險說明，不得解讀成正式交易決策。",
            "active_in_trade_decision": False,
            "components": ["three_pool_vote_diagnostic", "pool3_shadow_or_diagnostic", "final_decision_layer_report_only", "chip_factor_shadow_diagnostic"],
        },
        "execution_boundary": {
            "label": "尚未成立",
            "description": "正式換倉與出場規則尚未建立；目前模型訊號不等於完整持倉管理命令。",
            "active_in_trade_decision": False,
        },
        "plain_language_notes": [
            "正式模型基準：目前以主攻池提出觀察標的，確認池負責做風險確認。",
            "診斷註解：非正式診斷資訊只解釋風險或資料限制，不是正式交易規則。",
            "執行邊界：目前沒有完整換倉與出場層；訊號變化不等於完整持倉管理命令。",
        ],
    }


def _decision_first_report_contract(manifest: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, Any]:
    visible_generated = [
        item for item in manifest.get("generated", []) if not _hide_from_formal_report(item)
    ]
    active_rows = [item for item in visible_generated if item.get("active_in_trade_decision")]
    primary = active_rows[0] if active_rows else (visible_generated[0] if visible_generated else {})
    target = next((row for row in primary.get("top_candidates") or [] if row.get("is_model_target")), None)
    target = target or ((primary.get("top_candidates") or [{}])[0] if primary.get("top_candidates") else {})
    target_display = str(target.get("display") or primary.get("top_display") or primary.get("top_ticker") or "")
    target_ticker = str(target.get("ticker") or primary.get("top_ticker") or "")
    report_ready = bool(manifest.get("formal_report_ready", True))
    blockers = manifest.get("formal_report_blockers") or []
    if not report_ready:
        state = "data_blocked"
        conclusion = "資料尚未補齊，這份報告不能作為隔天操作判斷。"
    elif target_display:
        state = "formal_target_available"
        conclusion = f"正式觀察標的：{target_display}。"
    else:
        state = "no_formal_target"
        conclusion = "本次沒有形成正式觀察標的。"
    previous = _previous_formal_target_contract(manifest, output_root=output_root)
    switch_state, switch_wording = _switch_signal_state(
        current_ticker=target_ticker,
        current_display=target_display,
        previous=previous,
        report_ready=report_ready,
    )
    return {
        "decision_first_state": state,
        "decision_first_conclusion_zh": conclusion,
        "formal_target_display": target_display,
        "formal_target_ticker": target_ticker,
        "previous_formal_target_date": previous.get("previous_formal_target_date", ""),
        "previous_formal_target_display": previous.get("previous_formal_target_display", ""),
        "previous_formal_target_ticker": previous.get("previous_formal_target_ticker", ""),
        "data_completeness_state": "complete" if report_ready else "blocked",
        "data_blocker_summary_zh": "；".join(str(item.get("reason_zh") or item.get("reason") or "") for item in blockers if item) if blockers else "",
        "switch_signal_state": switch_state,
        "switch_signal_wording_zh": switch_wording,
        "score_margin_state": "formal_candidate_ranking_contract_missing",
        "score_margin_wording_zh": "正式候選排名與第二、第三名分數差距契約尚未完成，不能用 proxy 分數冒充正式分數差距。",
        "pool1_state_zh": _sanitize_visible_report_reason(primary.get("selection_reason") or primary.get("gate_reason") or "主攻池已產生正式觀察。") if primary else "主攻池未產生正式觀察。",
        "pool2_state_zh": "確認池風控已納入目前正式模型；更細的確認品質需等待正式候選排名契約補齊。",
        "active_in_trade_decision": False,
        "boundary": "report_contract",
    }


def _previous_formal_target_contract(manifest: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, str]:
    current_date = str(manifest.get("actual_signal_date") or manifest.get("signal_date") or "")
    if not output_root:
        return {}
    base = Path(output_root)
    if not base.exists():
        return {}
    candidates: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for path in base.glob("*/stock_pool_observation_manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not payload.get("formal_report_ready", True):
            continue
        signal_date = str(payload.get("actual_signal_date") or payload.get("signal_date") or "")
        if not signal_date or signal_date >= current_date:
            continue
        target = _extract_formal_target(payload)
        if not target.get("formal_target_ticker"):
            continue
        try:
            candidates.append((pd.Timestamp(signal_date), {"date": signal_date, **target}))
        except Exception:
            continue
    if not candidates:
        return {}
    _, latest = max(candidates, key=lambda item: item[0])
    return {
        "previous_formal_target_date": str(latest.get("date") or ""),
        "previous_formal_target_display": str(latest.get("formal_target_display") or ""),
        "previous_formal_target_ticker": str(latest.get("formal_target_ticker") or ""),
    }


def _extract_formal_target(manifest: dict[str, Any]) -> dict[str, str]:
    decision = manifest.get("decision_first_report_contract")
    if isinstance(decision, dict) and decision.get("formal_target_ticker"):
        return {
            "formal_target_display": str(decision.get("formal_target_display") or decision.get("formal_target_ticker") or ""),
            "formal_target_ticker": str(decision.get("formal_target_ticker") or ""),
        }
    visible_generated = [
        item for item in manifest.get("generated", []) if not _hide_from_formal_report(item)
    ]
    active_rows = [item for item in visible_generated if item.get("active_in_trade_decision")]
    primary = active_rows[0] if active_rows else (visible_generated[0] if visible_generated else {})
    target = next((row for row in primary.get("top_candidates") or [] if row.get("is_model_target")), None)
    target = target or ((primary.get("top_candidates") or [{}])[0] if primary.get("top_candidates") else {})
    return {
        "formal_target_display": str(target.get("display") or primary.get("top_display") or primary.get("top_ticker") or ""),
        "formal_target_ticker": str(target.get("ticker") or primary.get("top_ticker") or ""),
    }


def _switch_signal_state(
    *,
    current_ticker: str,
    current_display: str,
    previous: dict[str, str],
    report_ready: bool,
) -> tuple[str, str]:
    if not report_ready:
        return "data_blocked", "資料尚未補齊，不能判定今日是否形成換倉訊號。"
    previous_ticker = str(previous.get("previous_formal_target_ticker") or "")
    previous_display = str(previous.get("previous_formal_target_display") or previous_ticker or "")
    previous_date = str(previous.get("previous_formal_target_date") or "")
    if not current_ticker:
        return "no_formal_target", "今日沒有形成正式標的，因此沒有換倉訊號。"
    if not previous_ticker:
        return "previous_target_missing", "找不到前一份已完成正式報告，因此只能顯示今日正式標的，不能比較是否換倉。"
    if current_ticker == previous_ticker:
        return "maintain_formal_target", f"今日正式標的仍是 {current_display or current_ticker}，相對前一份正式報告（{previous_date}）維持不變。"
    return (
        "formal_target_changed",
        f"正式目標已從 {previous_display}（{previous_date}）轉向 {current_display or current_ticker}；這是模型目標轉向，是否執行換倉仍需看執行層與人工確認。",
    )


def _cashflow_report_boundary() -> dict[str, Any]:
    return {
        "cashflow_objective_capital_twd": 4_000_000,
        "cashflow_monthly_target_twd": 150_000,
        "cashflow_target_source": "user_updated_2026_06_27",
        "cashflow_model_objective_state": "growth_model_not_fixed_income",
        "cashflow_policy_state": "cashflow_objective_requires_cash_buffer",
        "cashflow_policy_reference": "capped_profit_withdrawal_150k",
        "cashflow_secondary_policy_reference": "drawdown_pause_withdrawal_10_cap_150k",
        "cashflow_cash_buffer_required_twd": 2_400_000,
        "cashflow_shortfall_months_reference": 16,
        "cashflow_target_hit_rate_reference": 0.4528,
        "cashflow_longest_under_target_reference": 16,
        "cashflow_capped_profit_total_withdrawal_twd": 3_600_000,
        "cashflow_capped_profit_final_equity_twd": 67_420_000,
        "cashflow_partial_profit_total_withdrawal_twd": 3_849_300,
        "cashflow_partial_profit_final_equity_twd": 63_690_000,
        "cashflow_drawdown_pause_hit_rate_reference": 0.4340,
        "cashflow_drawdown_pause_total_withdrawal_twd": 3_450_000,
        "cashflow_drawdown_pause_final_equity_twd": 68_260_000,
        "cashflow_fixed_150k_final_equity_twd": 2_556_900,
        "cashflow_fixed_150k_max_drawdown": -0.6932,
        "cashflow_account_depletion_warning": "legacy_200k_fixed_withdrawal_depleted_2023_12",
        "cashflow_legacy_stress_test_reference": "fixed_200k_every_month_depleted_2023_12",
        "cashflow_drawdown_pause_reference": "drawdown_pause_withdrawal_10_cap_150k",
        "cashflow_active_in_trade_decision": False,
        "cashflow_boundary": "report_only",
        "cashflow_wording_zh": (
            "模型目前定位是資產成長，不是固定月薪機器。新版現金流目標是400萬本金、月生活費上限15萬；"
            "若當月收益未達15萬，不應硬提到15萬。"
        ),
        "cashflow_reference_wording_zh": (
            "15萬目標比20萬高壓版更接近可執行，但歷史回測中完整領到15萬的月份約45%；"
            "若每月支出固定15萬且不靠硬提本金補缺口，至少需要約240萬外部生活費緩衝。"
        ),
        "cashflow_legacy_stress_wording_zh": "舊20萬高壓測試顯示，固定每月硬提20萬曾在2023-12歸零；此結果只作高壓提醒。",
        "cashflow_wording_policy": "現金流診斷不得寫成收入承諾或固定提款能力。",
    }


def _attach_cashflow_report_boundary(manifest: dict[str, Any]) -> None:
    cashflow = _cashflow_report_boundary()
    manifest["cashflow_report_boundary"] = cashflow
    for key in (
        "cashflow_objective_capital_twd",
        "cashflow_monthly_target_twd",
        "cashflow_target_source",
        "cashflow_policy_state",
        "cashflow_policy_reference",
        "cashflow_cash_buffer_required_twd",
        "cashflow_shortfall_months_reference",
        "cashflow_account_depletion_warning",
        "cashflow_target_hit_rate_reference",
        "cashflow_longest_under_target_reference",
        "cashflow_active_in_trade_decision",
        "cashflow_boundary",
    ):
        manifest[key] = cashflow.get(key)
    manifest["formal_model_changed"] = False
    manifest["trade_decision_changed"] = False


def _target_stability_warning_boundary(manifest: dict[str, Any]) -> dict[str, Any]:
    explicit = manifest.get("target_stability_warning")
    if isinstance(explicit, dict):
        return explicit
    generated = manifest.get("generated") or []
    formal_rows = [
        row for row in generated
        if not _hide_from_formal_report(
            {
                "pool_id": row.get("pool_id", ""),
                "pool_name": row.get("pool_name", ""),
                "pool_short_name": _short_pool_name(row),
                "role_name": row.get("role_name", ""),
                "role_description": row.get("role_description", ""),
            }
        )
    ]
    primary = next((row for row in formal_rows if bool(row.get("active_in_trade_decision", False))), formal_rows[0] if formal_rows else {})
    top_candidates = primary.get("top_candidates") or []
    score_gap = _top_score_gap(top_candidates)
    low_score_gap = score_gap is not None and score_gap < TARGET_STABILITY_LOW_SCORE_GAP_THRESHOLD
    target_drop = _bool_like(manifest.get("target_drop_from_top3_3d_proxy_flag"))
    new_target_age_1 = _bool_like(manifest.get("new_target_confirmation_age_1_flag"))
    proxy_available = manifest.get("target_drop_from_top3_3d_proxy_flag") is not None
    if not formal_rows:
        state = "data_insufficient"
        reason = "正式觀察資料不足，無法判斷標的穩定度。"
    elif target_drop:
        state = "target_stability_watch"
        reason = "診斷 proxy 顯示目前標的在三個交易日內掉出前三名，穩定度需觀察。"
    elif low_score_gap:
        state = "low_score_margin_watch"
        reason = f"第一名與第二名分數差距約 {_format_score_gap(score_gap)}，優勢不夠明顯。"
    elif new_target_age_1:
        state = "new_target_unconfirmed_watch"
        reason = "新標的剛出現，尚未累積足夠確認天數。"
    elif not proxy_available:
        state = "mixed_or_insufficient"
        reason = "每日正式報告尚未接入短期掉出前三名的診斷來源；目前只顯示分數差距與新訊號觀察。"
    else:
        state = "stable_target"
        reason = "目前沒有偵測到分數差距偏低或短期掉出前三名 proxy 警示。"
    return {
        "target_stability_warning_state": state,
        "target_stability_warning_reason": reason,
        "target_drop_from_top3_3d_proxy_flag": bool(target_drop),
        "target_drop_from_top3_3d_proxy_available": bool(proxy_available),
        "low_score_gap_proxy_flag": bool(low_score_gap),
        "new_target_confirmation_age_1_flag": bool(new_target_age_1),
        "pool2_disagreement_negative_warning_used": False,
        "any_low_confidence_warning_used": False,
        "target_score_gap_to_second": score_gap,
        "target_stability_warning_active_in_trade_decision": False,
        "target_stability_warning_boundary": "report_only",
        "target_stability_proxy_contract": (
            "短期掉出前三名目前來自重建的分數排序診斷，"
            "不是正式候選排序契約；未建立正式分數差距資料契約前不得升級為交易規則。"
        ),
    }


def _attach_target_stability_warning_boundary(manifest: dict[str, Any]) -> None:
    warning = _target_stability_warning_boundary(manifest)
    manifest["target_stability_warning"] = warning
    for key in (
        "target_stability_warning_state",
        "target_stability_warning_reason",
        "target_drop_from_top3_3d_proxy_flag",
        "low_score_gap_proxy_flag",
        "new_target_confirmation_age_1_flag",
        "target_stability_warning_active_in_trade_decision",
        "target_stability_warning_boundary",
        "target_stability_proxy_contract",
    ):
        manifest[key] = warning.get(key)
    manifest["formal_model_changed"] = False
    manifest["trade_decision_changed"] = False
    manifest["active_in_trade_decision"] = False


def _live_risk_regime_warning_boundary(manifest: dict[str, Any]) -> dict[str, Any]:
    explicit = manifest.get("live_risk_regime_warning")
    if isinstance(explicit, dict):
        return explicit
    generated = manifest.get("generated") or []
    formal_rows = [
        row for row in generated
        if not _hide_from_formal_report(
            {
                "pool_id": row.get("pool_id", ""),
                "pool_name": row.get("pool_name", ""),
                "pool_short_name": _short_pool_name(row),
                "role_name": row.get("role_name", ""),
                "role_description": row.get("role_description", ""),
            }
        )
    ]
    active_rows = [row for row in formal_rows if bool(row.get("active_in_trade_decision", False))]
    primary = active_rows[0] if active_rows else (formal_rows[0] if formal_rows else {})
    metadata = primary.get("source_metadata") if isinstance(primary.get("source_metadata"), dict) else {}
    risk_off_active = _bool_like(metadata.get("risk_off_active"))
    attack_gate_active = _bool_like(metadata.get("attack_gate_active"))
    market_regime_label = str(metadata.get("market_regime_label") or "").strip()
    data_end_date = str(primary.get("data_end_date") or manifest.get("actual_signal_date") or manifest.get("signal_date") or "")
    if not formal_rows:
        state = "data_insufficient"
        reason = "正式觀察資料不足，無法判斷目前市場風險環境。"
    elif risk_off_active:
        state = "risk_off_watch"
        reason = "正式訊號顯示市場環境偏防守，歷史診斷中這類狀態的回撤壓力較高。"
    elif not attack_gate_active:
        state = "weak_regime_watch"
        reason = "主攻條件尚未完全打開，市場環境偏弱，需留意回撤與月度現金流壓力。"
    else:
        state = "risk_on"
        reason = "目前主攻條件維持開啟，尚未偵測到正式風險提醒。"
    if market_regime_label:
        source = f"正式主攻訊號、市場狀態={_translate_internal_visible_text(market_regime_label)}"
    else:
        source = "正式主攻訊號的攻擊條件與風險狀態"
    return {
        "live_risk_regime_state": state,
        "live_risk_regime_warning_reason": reason,
        "live_risk_regime_feature_source": source,
        "live_risk_regime_data_end_date": data_end_date,
        "live_risk_regime_active_in_trade_decision": False,
        "live_risk_regime_boundary": "report_only",
        "live_risk_regime_breadth_readiness": "breadth_not_ready",
        "live_risk_regime_throttle_diagnostic_note": (
            "曝險縮放只停留在歷史診斷；目前沒有啟用正式降曝險規則，也不改變正式標的。"
        ),
    }


def _attach_live_risk_regime_warning_boundary(manifest: dict[str, Any]) -> None:
    warning = _live_risk_regime_warning_boundary(manifest)
    manifest["live_risk_regime_warning"] = warning
    for key in (
        "live_risk_regime_state",
        "live_risk_regime_warning_reason",
        "live_risk_regime_feature_source",
        "live_risk_regime_data_end_date",
        "live_risk_regime_active_in_trade_decision",
        "live_risk_regime_boundary",
        "live_risk_regime_breadth_readiness",
    ):
        manifest[key] = warning.get(key)
    manifest["formal_model_changed"] = False
    manifest["trade_decision_changed"] = False
    manifest["active_in_trade_decision"] = False


def _chip_context_report_boundary(manifest: dict[str, Any]) -> dict[str, Any]:
    explicit = manifest.get("chip_context")
    if isinstance(explicit, dict):
        return explicit
    signal_date = str(manifest.get("actual_signal_date") or manifest.get("signal_date") or "")
    generated = manifest.get("generated") or []
    formal_rows = [
        row for row in generated
        if not _hide_from_formal_report(
            {
                "pool_id": row.get("pool_id", ""),
                "pool_name": row.get("pool_name", ""),
                "pool_short_name": _short_pool_name(row),
                "role_name": row.get("role_name", ""),
                "role_description": row.get("role_description", ""),
            }
        )
    ]
    active_rows = [row for row in formal_rows if bool(row.get("active_in_trade_decision", False))]
    primary = active_rows[0] if active_rows else (formal_rows[0] if formal_rows else {})
    target = next((row for row in primary.get("top_candidates") or [] if row.get("is_model_target")), None)
    target = target or ((primary.get("top_candidates") or [{}])[0] if primary.get("top_candidates") else {})
    h1_positive = _number_like(target.get("bullish_flow_score")) > 0
    h2_sell_pressure = _number_like(target.get("institutional_risk")) > 0 or _number_like(target.get("flow_risk_score")) > 0
    coverage_end = _chip_context_coverage_end(manifest, target)
    if not formal_rows:
        state = "chip_not_available"
        reason = "正式觀察資料不足，無法建立籌碼輔助觀察。"
    elif not coverage_end:
        state = "chip_data_insufficient"
        reason = "本次正式產報沒有成功載入當日三大法人籌碼資料，應先補齊資料後再發布。"
        h1_positive = False
        h2_sell_pressure = False
    elif _date_after(signal_date, coverage_end):
        state = "chip_data_insufficient"
        reason = (
            f"目前籌碼資料只更新到 {coverage_end}，尚未補到本次正式訊號日 {signal_date}，"
            f"應先補齊資料後再發布正式報告。"
        )
        h1_positive = False
        h2_sell_pressure = False
    elif h1_positive and h2_sell_pressure:
        state = "mixed_chip_context"
        reason = "法人籌碼同時出現偏正向與賣壓訊號，只作輔助觀察。"
    elif h1_positive:
        state = "h1_positive_context"
        reason = "法人籌碼偏正向，僅作輔助觀察，不提高正式權重。"
    elif h2_sell_pressure:
        state = "h2_sell_pressure_observation"
        reason = "籌碼賣壓存在，但歷史診斷不支持作為正式否決或降權。"
    else:
        state = "chip_data_insufficient"
        reason = "目前沒有乾淨中性對照組，籌碼資料只作背景觀察。"
    return {
        "chip_context_state": state,
        "chip_context_reason": reason,
        "chip_h1_positive_flag": bool(h1_positive),
        "chip_h2_sell_pressure_flag": bool(h2_sell_pressure),
        "chip_neutral_reference_available": False,
        "chip_data_coverage_end": coverage_end,
        "chip_context_active_in_trade_decision": False,
        "chip_context_boundary": "report_only",
        "chip_context_policy_note": (
            "法人籌碼偏正向只能作輔助觀察；籌碼賣壓不作正式否決、降權或換倉規則。"
        ),
    }


def _attach_chip_context_report_boundary(manifest: dict[str, Any]) -> None:
    context = _chip_context_report_boundary(manifest)
    manifest["chip_context"] = context
    for key in (
        "chip_context_state",
        "chip_context_reason",
        "chip_h1_positive_flag",
        "chip_h2_sell_pressure_flag",
        "chip_neutral_reference_available",
        "chip_data_coverage_end",
        "chip_context_active_in_trade_decision",
        "chip_context_boundary",
    ):
        manifest[key] = context.get(key)
    manifest["formal_model_changed"] = False
    manifest["trade_decision_changed"] = False
    manifest["active_in_trade_decision"] = False


def _live_risk_regime_state_label(value: object) -> str:
    mapping = {
        "risk_on": "市場環境偏強",
        "weak_regime_watch": "市場環境偏弱，留意風險",
        "risk_off_watch": "防守環境觀察",
        "data_insufficient": "資料不足",
        "breadth_not_ready": "市場廣度資料尚未納入正式契約",
    }
    return mapping.get(str(value or ""), str(value or "資料不足"))


def _chip_context_state_label(value: object) -> str:
    mapping = {
        "h1_positive_context": "法人籌碼偏正向",
        "h2_sell_pressure_observation": "籌碼賣壓觀察",
        "mixed_chip_context": "籌碼訊號混合",
        "chip_data_insufficient": "籌碼資料不足",
        "chip_not_available": "籌碼資料未接入",
    }
    return mapping.get(str(value or ""), str(value or "籌碼資料不足"))


def _risk_factor_coverage_end_by_kind(risk_signals: dict[str, RiskFactorSignal]) -> dict[str, str]:
    dates_by_kind: dict[str, list[str]] = {}
    for signal in risk_signals.values():
        for kind in signal.source_kinds:
            dates_by_kind.setdefault(kind, []).extend(signal.source_dates)
    coverage: dict[str, str] = {}
    for kind, dates in dates_by_kind.items():
        valid = sorted({str(date) for date in dates if date})
        if valid:
            coverage[kind] = valid[-1]
    return coverage


def _chip_context_coverage_end(manifest: dict[str, Any], target: dict[str, Any]) -> str:
    by_kind = manifest.get("risk_factor_coverage_end_by_kind")
    if isinstance(by_kind, dict) and by_kind.get("institutional"):
        return str(by_kind["institutional"])
    source_dates = target.get("flow_source_dates")
    if isinstance(source_dates, (list, tuple)):
        valid = sorted(str(item) for item in source_dates if item)
        return valid[-1] if valid else ""
    if isinstance(source_dates, str) and source_dates:
        valid = sorted(item.strip() for item in source_dates.split(",") if item.strip())
        return valid[-1] if valid else ""
    return ""


def _number_like(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _date_after(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return pd.Timestamp(left).normalize() > pd.Timestamp(right).normalize()
    except (TypeError, ValueError):
        return False


def _top_score_gap(candidates: list[dict[str, Any]]) -> float | None:
    scores = []
    for row in candidates[:2]:
        try:
            scores.append(float(row.get("score") or row.get("rank_score") or 0))
        except (TypeError, ValueError):
            scores.append(0.0)
    if len(scores) < 2:
        return None
    return round(scores[0] - scores[1], 6)


def _format_score_gap(value: float | None) -> str:
    if value is None:
        return "資料不足"
    return f"{value:.4f}"


def _target_stability_state_label(value: object) -> str:
    mapping = {
        "stable_target": "目前未見明顯穩定度警示",
        "target_stability_watch": "短期排名穩定度需觀察",
        "low_score_margin_watch": "分數優勢偏低",
        "new_target_unconfirmed_watch": "新標的尚未確認",
        "mixed_or_insufficient": "診斷資料不足，僅作觀察",
        "data_insufficient": "資料不足",
    }
    return mapping.get(str(value or ""), str(value or "資料不足"))


def _bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "是"}


def write_stock_pool_observation_batch_summary(root: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for item in manifest.get("generated", []):
        rows.append(
            {
                "status": "generated",
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "pool_short_name": _short_pool_name(item),
                "role_name": item.get("role_name", ""),
                "role_description": item.get("role_description", ""),
                "candidate_review_frequency": item.get("candidate_review_frequency", ""),
                "candidate_update_policy": item.get("candidate_update_policy", ""),
                "signal_date": item.get("signal_date", manifest.get("signal_date", "")),
                "top_display": item.get("top_display", ""),
                "top_ticker": item.get("top_ticker", ""),
                "top_asset_type": item.get("top_asset_type", ""),
                "score": item.get("score", ""),
                "rank_score": item.get("rank_score", item.get("score", "")),
                "rank": item.get("rank", ""),
                "base_pool_passed": item.get("base_pool_passed", ""),
                "attack_gate_open": item.get("attack_gate_open", ""),
                "eligible_for_pool_selection": item.get("eligible_for_pool_selection", False),
                "selection_layer": item.get("selection_layer", ""),
                "selection_reason": item.get("selection_reason", ""),
                "gate_rule_id": item.get("gate_rule_id", ""),
                "gate_reason": item.get("gate_reason", ""),
                "action_state": item.get("action_state", ""),
                "decision_layer": item.get("decision_layer", ""),
                "active_in_trade_decision": item.get("active_in_trade_decision", False),
                "source_module": item.get("source_module", ""),
                "report_line": (item.get("dispatch") or {}).get("report_line", ""),
                "workflow_file": (item.get("dispatch") or {}).get("workflow_file", ""),
                "missing_price_tickers": ",".join(_visible_missing_price_tickers(item.get("missing_price_tickers") or [])),
                "source_summary": _source_summary(item.get("source_metadata") or {}),
                "top_candidates": item.get("top_candidates") or [],
                "top_candidates_text": _top_candidates_text(item.get("top_candidates") or []),
                "reason": "",
                "output_dir": item.get("output_dir", ""),
            }
        )
    for item in manifest.get("skipped", []):
        rows.append(
            {
                "status": "skipped",
                "pool_id": item.get("pool_id", ""),
                "pool_name": item.get("pool_name", ""),
                "pool_short_name": _short_pool_name(item),
                "role_name": item.get("role_name", ""),
                "role_description": item.get("role_description", ""),
                "candidate_review_frequency": item.get("candidate_review_frequency", ""),
                "candidate_update_policy": item.get("candidate_update_policy", ""),
                "signal_date": manifest.get("signal_date", ""),
                "top_display": "",
                "top_ticker": "",
                "top_asset_type": "",
                "score": "",
                "rank_score": "",
                "rank": "",
                "base_pool_passed": False,
                "attack_gate_open": "",
                "eligible_for_pool_selection": False,
                "selection_layer": SELECTION_NO_SELECTION,
                "selection_reason": _sanitize_visible_report_reason(item.get("reason", "")),
                "gate_rule_id": "",
                "gate_reason": _sanitize_visible_report_reason(item.get("reason", "")),
                "action_state": "",
                "decision_layer": item.get("decision_layer", ""),
                "active_in_trade_decision": item.get("active_in_trade_decision", False),
                "source_module": item.get("source_module", ""),
                "report_line": (item.get("dispatch") or {}).get("report_line", ""),
                "workflow_file": (item.get("dispatch") or {}).get("workflow_file", ""),
                "missing_price_tickers": "",
                "source_summary": "",
                "top_candidates": [],
                "top_candidates_text": "",
                "reason": _sanitize_visible_report_reason(item.get("reason", "")),
                "output_dir": "",
            }
        )
    pd.DataFrame(rows).to_csv(root / "stock_pool_observation_summary.csv", index=False, encoding="utf-8-sig")
    (root / "stock_pool_observation_report.md").write_text(
        markdown_observation_batch_report(manifest, rows),
        encoding="utf-8",
    )
    if manifest.get("generated") and manifest.get("formal_report_ready", True):
        write_stock_pool_observation_batch_pdf(root / REPORT_LATEST_FILENAME, manifest, rows)


def _manifest_pool_candidate_tickers(manifest: dict[str, Any], pool_id: str) -> set[str]:
    result: set[str] = set()
    for item in manifest.get("generated", []):
        if item.get("pool_id") != pool_id:
            continue
        if item.get("top_ticker"):
            result.add(str(item["top_ticker"]))
        for row in item.get("top_candidates") or []:
            ticker = str(row.get("ticker") or "").strip()
            if ticker:
                result.add(ticker)
    return result


def _visible_missing_price_tickers(tickers: list[str]) -> list[str]:
    return [
        str(ticker)
        for ticker in tickers
        if _normalize_ticker(str(ticker)) not in FORMAL_CANDIDATE_EXCLUDED_TICKERS
    ]


def _sanitize_visible_report_reason(reason: object) -> str:
    text = str(reason or "")
    prefix = "No price data available for pool tickers:"
    if text == "missing_formal_radar_candidates":
        return "未納入正式結論：非正式觀察區目前沒有可用候選。"
    if text == "no_resolved_symbols":
        return "資料不足：這個觀察區目前沒有可用成員。"
    if text.startswith("No exact common price data"):
        return "資料不足：共同價格資料不完整，無法產生正式觀察。"
    market_data_match = re.search(r"No market data for signal date ([0-9-]+); latest available is ([0-9-]+)", text)
    if market_data_match:
        requested, latest = market_data_match.groups()
        return f"資料不足：{requested} 還沒有完整市場資料，目前可用到 {latest}。"
    if not text.startswith(prefix):
        return _translate_internal_visible_text(text)
    raw = text[len(prefix):].strip()
    tickers = [item.strip() for item in raw.split(",") if item.strip()]
    visible = _visible_missing_price_tickers(tickers)
    if not visible:
        return "資料不足：正式候選沒有可用價格資料。"
    return f"資料不足：缺少價格資料（{', '.join(visible)}）。"


def _user_facing_candidate_reason(reason: object) -> str:
    text = _sanitize_visible_report_reason(reason)
    text = text.replace("大型廣度池 v1：", "")
    text = text.replace("風格補強池 v1：", "")
    text = text.replace("通用池基礎 gate：", "")
    text = text.replace("通用池基礎 gate 未通過：", "基本條件未通過：")
    text = re.sub(r"base=Y", "基本條件通過", text)
    text = re.sub(r"base=N", "基本條件未通過", text)
    text = re.sub(r"=Y(?=；|。|$)", "：通過", text)
    text = re.sub(r"=N(?=；|。|$)", "：未通過", text)
    text = re.sub(r"\(Y\)", "，通過", text)
    text = re.sub(r"\(N\)", "，未通過", text)
    text = text.replace("20/60動能品質", "20日與60日動能品質")
    text = text.replace("60日相對0050超額", "60日表現相對0050")
    text = text.replace("60日相對0050強度", "60日表現相對0050")
    text = text.replace("60/120中期上攻力", "60日與120日中期上攻力")
    text = text.replace("120日機會成本", "120日相對0050機會成本")
    text = text.replace("20日回撤控管", "20日回撤風險")
    text = text.replace("籌碼風險", "籌碼風險分數")
    text = text.replace("benchmark", "基準")
    text = text.replace("ETF", "市場型 ETF")
    text = text.replace("gate", "門檻")
    return _translate_internal_visible_text(text)


def _translate_internal_visible_text(text: object) -> str:
    value = str(text or "")
    replacements = {
        "combined_cap40_confirmation1_base": "目前正式模型",
        "pool1_primary_pool2_confirmation_cap40": "主攻池優先、確認池風險確認",
        "Pool1+Pool2 formal baseline": "正式模型基準",
        "Pool1+Pool2": "主攻池 + 確認池",
        "PIT-ready Pool2": "已通過歷史成分檢查的確認池",
        "selector": "選股邏輯",
        "formal target": "正式採用版本",
        "正式 target": "正式採用版本",
        "baseline selection signal": "模型觀察訊號",
        "execution/exit layer": "換倉與出場層",
        "formal model target": "正式模型版本",
    }
    for raw, translated in replacements.items():
        value = value.replace(raw, translated)
    return value


def _skipped_reason_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    skipped_rows = [row for row in rows if row.get("status") != "generated"]
    visible = [row for row in skipped_rows if not _hide_from_formal_report(row)]
    hidden = [row for row in skipped_rows if _hide_from_formal_report(row)]
    summaries: list[str] = []
    for row in visible[:3]:
        pool_name = row.get("pool_short_name") or row.get("pool_name") or "未命名觀察區"
        summaries.append(f"{pool_name}：{_sanitize_visible_report_reason(row.get('reason'))}")
    if hidden:
        summaries.append(f"非正式觀察區：{len(hidden)} 個項目不列入正式報告主表。")
    if not summaries and manifest.get("skipped"):
        summaries.append("有項目暫時無法產生正式觀察，請查看機器可讀 manifest 取得完整內部原因。")
    return summaries


def markdown_observation_batch_report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    wording = manifest.get("report_wording_boundary") or _report_wording_boundary()
    formal_boundary = wording.get("formal_baseline") or {}
    cashflow = manifest.get("cashflow_report_boundary") or _cashflow_report_boundary()
    stability = manifest.get("target_stability_warning") or _target_stability_warning_boundary(manifest)
    live_risk = manifest.get("live_risk_regime_warning") or _live_risk_regime_warning_boundary(manifest)
    chip_context = manifest.get("chip_context") or _chip_context_report_boundary(manifest)
    decision_first = manifest.get("decision_first_report_contract") or _decision_first_report_contract(manifest)
    report_rows = _formal_report_rows(rows)
    visible_skipped_count = len([row for row in report_rows if row.get("status") != "generated"])
    skipped_summaries = _skipped_reason_summary(rows, manifest)
    report_ready = bool(manifest.get("formal_report_ready", True))
    lines = [
        "# 股票池觀察摘要",
        "",
        f"- 要求訊號日：{manifest.get('requested_signal_date', manifest.get('signal_date', ''))}",
        f"- 訊號日：{manifest.get('signal_date', '')}",
        f"- 資料日 fallback：{manifest.get('fallback_reason') or '未啟用'}",
        f"- 正式觀察項目：{len(report_rows)}",
        f"- 暫無正式觀察：{visible_skipped_count}（{'; '.join(skipped_summaries) if skipped_summaries else '無'}）",
        f"- 正式報告狀態：{'可發布' if report_ready else '停止發布，等待完整資料'}",
        f"- 正式模型基準：{_translate_internal_visible_text(formal_boundary.get('description'))}",
        "",
        "## 隔天操作判斷",
        "",
        f"- 主結論：{decision_first.get('decision_first_conclusion_zh', '')}",
        f"- 正式標的：{decision_first.get('formal_target_display') or '無'}",
        f"- 前一份正式報告標的：{decision_first.get('previous_formal_target_display') or '無'}"
        f"{'（' + str(decision_first.get('previous_formal_target_date')) + '）' if decision_first.get('previous_formal_target_date') else ''}",
        f"- 資料完整度：{'資料齊全' if decision_first.get('data_completeness_state') == 'complete' else '資料未補齊'}",
        f"- 資料缺口：{decision_first.get('data_blocker_summary_zh') or '無'}",
        f"- 換倉訊號：{decision_first.get('switch_signal_wording_zh', '')}",
        f"- 分數差距：{decision_first.get('score_margin_wording_zh', '')}",
        f"- 主攻池狀態：{decision_first.get('pool1_state_zh', '')}",
        f"- 確認池狀態：{decision_first.get('pool2_state_zh', '')}",
        "",
        "## 現金流健康度（僅供診斷）",
        "",
        f"- 目標本金：{_format_twd(cashflow.get('cashflow_objective_capital_twd'))}",
        f"- 月生活費目標上限：{_format_twd(cashflow.get('cashflow_monthly_target_twd'))}",
        f"- 目前定位：{cashflow.get('cashflow_wording_zh', '')}",
        f"- 歷史達標參考：完整領到15萬的月份約{float(cashflow.get('cashflow_target_hit_rate_reference') or 0) * 100:.2f}%",
        f"- 外部現金緩衝需求：{_format_twd(cashflow.get('cashflow_cash_buffer_required_twd'))}，用來支付連續短缺期。",
        f"- 診斷說明：{cashflow.get('cashflow_reference_wording_zh', '')}",
        f"- 舊高壓測試：{cashflow.get('cashflow_legacy_stress_wording_zh', '')}",
        f"- 邊界：僅供報告參考，不改正式模型、不改交易決策、不代表提款指令。",
        "",
        "## 標的穩定度提醒（僅供診斷）",
        "",
        f"- 狀態：{_target_stability_state_label(stability.get('target_stability_warning_state'))}",
        f"- 原因：{stability.get('target_stability_warning_reason', '')}",
        f"- 診斷來源邊界：{stability.get('target_stability_proxy_contract', '')}",
        f"- 邊界：僅供報告提醒，不改正式模型、不改正式標的、不代表換倉指令。",
        "",
        "## 市場風險環境提醒（僅供診斷）",
        "",
        f"- 狀態：{_live_risk_regime_state_label(live_risk.get('live_risk_regime_state'))}",
        f"- 原因：{live_risk.get('live_risk_regime_warning_reason', '')}",
        f"- 資料截止日：{live_risk.get('live_risk_regime_data_end_date', '')}",
        f"- 資料來源：{live_risk.get('live_risk_regime_feature_source', '')}",
        f"- 市場廣度資料狀態：{_live_risk_regime_state_label(live_risk.get('live_risk_regime_breadth_readiness'))}",
        f"- 診斷說明：{live_risk.get('live_risk_regime_throttle_diagnostic_note', '')}",
        f"- 提醒：僅供報告提醒，不改正式模型、不改正式標的、不代表交易指令。",
        "",
        "## 籌碼背景觀察（僅供診斷）",
        "",
        f"- 狀態：{_chip_context_state_label(chip_context.get('chip_context_state'))}",
        f"- 原因：{chip_context.get('chip_context_reason', '')}",
        f"- 籌碼資料截止日：{chip_context.get('chip_data_coverage_end', '')}",
        f"- 中性對照組：{'可用' if chip_context.get('chip_neutral_reference_available') else '目前不可用'}",
        f"- 診斷說明：{chip_context.get('chip_context_policy_note', '')}",
        f"- 提醒：僅供報告提醒，不改正式模型、不改正式標的、不代表交易指令。",
        "",
        "| 狀態 | 正式觀察 | 角色 | 成員檢查 | 前三名 / 暫無觀察原因 | 來源摘要 | 缺價股票 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        if row["status"] == "generated":
            target = row["top_candidates_text"] or row["top_display"] or row["top_ticker"] or row["action_state"] or "無合格候選"
            missing = row["missing_price_tickers"] or "-"
            status = "已產生"
        else:
            target = row["reason"] or "skipped"
            missing = "-"
            status = "暫無觀察"
        role = row.get("role_name") or "-"
        review_frequency = _candidate_review_label(row.get("candidate_review_frequency"))
        lines.append(f"| {status} | {row.get('pool_short_name') or row['pool_name']} | {role} | {review_frequency} | {target} | {row.get('source_summary') or '-'} | {missing} |")
    return "\n".join(lines)


def write_stock_pool_observation_batch_pdf(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _configure_chinese_font()
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_observation_summary_pdf_page(ax, manifest, rows)
        _save_figure_as_raster_pdf_page(pdf, fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f6f8")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        _draw_observation_detail_pdf_page(ax, manifest, rows)
        _save_figure_as_raster_pdf_page(pdf, fig)


def _formal_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not _hide_from_formal_report(row)]


def _hide_from_formal_report(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("pool_id", "pool_name", "pool_short_name", "role_name", "role_description")
    )
    return any(marker in text for marker in ("large_core_bluechip_v0", "風格補強", "Pool3", "pool3", "Radar", "radar", "雷達"))


def _draw_observation_summary_pdf_page(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    report_rows = _formal_report_rows(rows)
    generated_count = len([row for row in report_rows if row.get("status") == "generated"])
    skipped_count = len([row for row in report_rows if row.get("status") != "generated"])
    skipped_summaries = _skipped_reason_summary(rows, manifest)
    report_ready = bool(manifest.get("formal_report_ready", True))

    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.94, REPORT_TITLE, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.895,
        f"訊號日 {manifest.get('signal_date', '')} · {REPORT_VERSION}",
        color="#c8d5df",
        fontsize=11,
        transform=ax.transAxes,
    )
    if manifest.get("signal_date_fallback_used"):
        ax.text(
            0.06,
            0.875,
            f"手動補跑資料日：要求 {manifest.get('requested_signal_date', '')}，實際使用 {manifest.get('actual_signal_date', '')}",
            color="#f6c177",
            fontsize=8.8,
            transform=ax.transAxes,
        )
    cards = [
        ("正式觀察項目", f"{generated_count}", "#13795b"),
        ("暫無正式觀察", f"{skipped_count}", "#b42318" if skipped_count else "#13795b"),
        ("正式模型", "主攻池+確認池", "#2457a7"),
        ("報告狀態", "可發布" if report_ready else "等待完整資料", "#13795b" if report_ready else "#b42318"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(
            plt.Rectangle((x, 0.74), 0.2, 0.085, facecolor="white", edgecolor="#d9e0e5", linewidth=1, transform=ax.transAxes)
        )
        ax.text(x + 0.014, 0.795, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.767, _compact_display(str(value), limit=18), color=color, fontsize=11.2, fontweight="bold", transform=ax.transAxes)

    ax.text(0.06, 0.69, "正式模型：主攻池提出觀察標的，確認池負責風險確認。", color="#52616b", fontsize=10, transform=ax.transAxes)
    if skipped_summaries:
        ax.text(0.06, 0.668, _compact_display("暫無正式觀察：" + "；".join(skipped_summaries), limit=78), color="#7a4b00", fontsize=8.4, transform=ax.transAxes)
    _draw_formal_baseline_panel(ax, manifest, report_rows)
    _draw_cashflow_report_panel(ax, manifest)
    ax.text(0.06, 0.026, f"{REPORT_TITLE} · {manifest.get('signal_date', '')}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.026, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _draw_observation_detail_pdf_page(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    report_rows = _formal_report_rows(rows)
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.958, "正式模型觀察明細", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.92,
        f"訊號日 {manifest.get('signal_date', '')} · 表格頁 · {REPORT_VERSION}",
        color="#c8d5df",
        fontsize=10.5,
        transform=ax.transAxes,
    )
    bottom_y = _draw_pool_top3_sections(ax, report_rows, start_y=0.84)
    reminder_y = max(min(bottom_y - 0.018, 0.075), 0.062)
    chip_context = manifest.get("chip_context") or _chip_context_report_boundary(manifest)
    chip_line = (
        f"籌碼背景：{_chip_context_state_label(chip_context.get('chip_context_state'))}；"
        f"{chip_context.get('chip_context_reason', '')}"
    )
    _draw_wrapped_text(
        ax,
        0.06,
        reminder_y + 0.018,
        chip_line,
        max_units=86,
        line_gap=0.012,
        color="#7a4b00",
        fontsize=7.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        reminder_y,
        "提醒：表格列出池內排序與原因；正式模型仍以主攻池 + 確認池為主，不代表完整換倉與出場層。",
        color="#52616b",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    ax.text(0.06, 0.04, f"{REPORT_TITLE} · {manifest.get('signal_date', '')}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.04, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _draw_formal_baseline_panel(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    panel_x, panel_y, panel_w, panel_h = 0.065, 0.355, 0.87, 0.325
    ax.add_patch(
        plt.Rectangle(
            (panel_x, panel_y),
            panel_w,
            panel_h,
            facecolor="#fbfcfd",
            edgecolor="#dde7ee",
            linewidth=1.1,
            transform=ax.transAxes,
        )
    )
    ax.text(panel_x + 0.018, panel_y + panel_h - 0.032, "正式模型基準", color="#17212a", fontsize=12.0, fontweight="bold", transform=ax.transAxes)
    ax.text(panel_x + panel_w - 0.018, panel_y + panel_h - 0.032, "主攻池 + 確認池", color="#52616b", fontsize=9.2, ha="right", transform=ax.transAxes)

    ax.add_patch(
        plt.Rectangle(
            (panel_x + 0.03, panel_y + 0.19),
            panel_w - 0.06,
            0.075,
            facecolor="#17212a",
            edgecolor="#2457a7",
            linewidth=1.8,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    ax.add_patch(plt.Rectangle((panel_x + 0.03, panel_y + 0.19), 0.012, 0.075, facecolor="#2457a7", edgecolor="#2457a7", transform=ax.transAxes, zorder=4))
    ax.text(panel_x + 0.055, panel_y + 0.236, "目前正式模型", color="#c8d5df", fontsize=8.8, transform=ax.transAxes, zorder=4)
    ax.text(panel_x + 0.055, panel_y + 0.209, "主攻池優先，確認池做風險確認", color="white", fontsize=12.0, fontweight="bold", transform=ax.transAxes, zorder=4)
    ax.text(panel_x + panel_w - 0.04, panel_y + 0.224, "0050正二上限40%；超出保留現金", color="#d6e1e8", fontsize=9.2, ha="right", transform=ax.transAxes, zorder=4)

    formal_rows = [row for row in rows if row["status"] == "generated"][:3]
    card_specs = [(0.095, 0.435), (0.385, 0.435), (0.675, 0.435)]
    for index, row in enumerate(formal_rows):
        x, y = card_specs[index]
        active = bool(row.get("active_in_trade_decision", False))
        color = "#13795b" if active else "#6b7780"
        fill = "#f4fbf8" if active else "#f8fafb"
        ax.add_patch(plt.Rectangle((x, y), 0.235, 0.108, facecolor=fill, edgecolor="#d5e0e6", linewidth=1.0, transform=ax.transAxes, zorder=2))
        ax.add_patch(plt.Rectangle((x, y), 0.012, 0.108, facecolor=color, edgecolor=color, transform=ax.transAxes, zorder=3))
        ax.text(
            x + 0.022,
            y + 0.078,
            _short_pool_name(row),
            color="#17212a",
            fontsize=9.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=4,
        )
        layer = "正式訊號" if active else "候選觀察"
        top_text = row.get("top_display") or row.get("top_ticker") or row.get("action_state") or "無合格候選"
        _draw_wrapped_text(
            ax,
            x + 0.022,
            y + 0.057,
            str(top_text),
            max_units=21,
            line_gap=0.017,
            color="#26323b",
            fontsize=8.2,
            transform=ax.transAxes,
            zorder=4,
        )
        ax.text(x + 0.022, y + 0.023, layer, color=color, fontsize=8.0, fontweight="bold", transform=ax.transAxes, zorder=4)

    ax.text(
        panel_x + 0.03,
        panel_y + 0.045,
        "正式報告以最新主攻池 + 確認池模型結果為主；其他診斷資訊不作正式交易規則。",
        color="#52616b",
        fontsize=8.7,
        transform=ax.transAxes,
    )
    stability = manifest.get("target_stability_warning") or _target_stability_warning_boundary(manifest)
    warning_text = f"標的穩定度：{_target_stability_state_label(stability.get('target_stability_warning_state'))}；{stability.get('target_stability_warning_reason', '')}"
    _draw_wrapped_text(
        ax,
        panel_x + 0.03,
        panel_y + 0.022,
        warning_text,
        max_units=86,
        line_gap=0.013,
        color="#7a4b00" if stability.get("target_stability_warning_state") != "stable_target" else "#52616b",
        fontsize=7.7,
        transform=ax.transAxes,
    )
    live_risk = manifest.get("live_risk_regime_warning") or _live_risk_regime_warning_boundary(manifest)
    risk_text = (
        f"市場環境：{_live_risk_regime_state_label(live_risk.get('live_risk_regime_state'))}；"
        f"{live_risk.get('live_risk_regime_warning_reason', '')}"
    )
    _draw_wrapped_text(
        ax,
        panel_x + 0.03,
        panel_y + 0.006,
        risk_text,
        max_units=86,
        line_gap=0.013,
        color="#7a4b00" if live_risk.get("live_risk_regime_state") != "risk_on" else "#52616b",
        fontsize=7.5,
        transform=ax.transAxes,
    )


def _draw_cashflow_report_panel(ax, manifest: dict[str, Any]) -> None:
    cashflow = manifest.get("cashflow_report_boundary") or _cashflow_report_boundary()
    x, y, w, h = 0.065, 0.205, 0.87, 0.115
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor="#fffaf0",
            edgecolor="#ead7a3",
            linewidth=1.0,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.018, y + h - 0.029, "現金流健康度（報告參考）", color="#17212a", fontsize=11.0, fontweight="bold", transform=ax.transAxes)
    ax.text(
        x + w - 0.018,
        y + h - 0.029,
        f"本金 {_format_twd(cashflow.get('cashflow_objective_capital_twd'))}｜月目標上限 {_format_twd(cashflow.get('cashflow_monthly_target_twd'))}",
        color="#7a4b00",
        fontsize=8.3,
        ha="right",
        transform=ax.transAxes,
    )
    _draw_wrapped_text(
        ax,
        x + 0.018,
        y + 0.058,
        "模型偏資產成長，不是固定月薪；15萬目標約45%月份完整達標，固定生活費仍需約240萬外部緩衝。",
        max_units=78,
        line_gap=0.014,
        color="#26323b",
        fontsize=8.4,
        transform=ax.transAxes,
    )
    _draw_wrapped_text(
        ax,
        x + 0.018,
        y + 0.026,
        str(cashflow.get("cashflow_legacy_stress_wording_zh") or ""),
        max_units=78,
        line_gap=0.014,
        color="#7a4b00",
        fontsize=7.6,
        transform=ax.transAxes,
    )


def _draw_pool_top3_sections(ax, rows: list[dict[str, Any]], *, start_y: float = 0.635) -> float:
    x0 = 0.06
    y = start_y
    generated_rows = [row for row in rows if row["status"] == "generated"]
    skipped_rows = [row for row in rows if row["status"] != "generated"]
    for row in generated_rows[:3]:
        ax.add_patch(plt.Rectangle((x0, y - 0.021), 0.88, 0.032, facecolor="#e9f0f5", edgecolor="#d7e0e7", transform=ax.transAxes))
        title = row.get("pool_short_name") or _short_pool_name(row)
        ax.text(x0 + 0.012, y - 0.011, title, color="#17212a", fontsize=10.8, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.93,
            y - 0.011,
            f"資料日 {row['signal_date']}",
            color="#66737d",
            fontsize=7.8,
            ha="right",
            transform=ax.transAxes,
        )
        y -= 0.038
        review_label = _candidate_review_label(row.get("candidate_review_frequency"))
        role_line = str(row.get("role_name") or "")
        if review_label != "-":
            role_line = f"{role_line}｜成員檢查：{review_label}" if role_line else f"成員檢查：{review_label}"
        if role_line:
            role_lines = _wrap_text_lines(role_line, max_units=54)
            _draw_wrapped_text(ax, x0 + 0.012, y + 0.008, role_line, max_units=54, line_gap=0.012, color="#52616b", fontsize=7.2, transform=ax.transAxes)
            y -= 0.012 * max(len(role_lines), 1)
        headers = ("排序", "層級", "標的", "分數", "程式判斷原因")
        widths = (0.06, 0.06, 0.19, 0.09, 0.48)
        if role_line:
            y -= 0.006
        header_h = 0.025
        ax.add_patch(plt.Rectangle((x0, y), 0.88, header_h, facecolor="#f7fafc", edgecolor="#e1e7ec", transform=ax.transAxes))
        x = x0
        for header, width in zip(headers, widths):
            ax.text(x + 0.008, y + 0.008, header, color="#52616b", fontsize=8.0, fontweight="bold", transform=ax.transAxes)
            x += width
        y -= 0.029
        for candidate in (row.get("top_candidates") or [])[:3]:
            fill = "#fff7e6" if candidate.get("is_model_target") else "white"
            display = _normalize_display_label(str(candidate.get("display", "")), str(candidate.get("ticker", "")))
            reason = _user_facing_candidate_reason(candidate.get("gate_reason") or candidate.get("reason", ""))
            reason_lines = _wrap_text_lines(reason, max_units=50)
            display_lines = _wrap_text_lines(display, max_units=18)
            row_h = max(0.034, 0.014 * max(len(reason_lines), len(display_lines), 1) + 0.018)
            ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor=fill, edgecolor="#e1e7ec", transform=ax.transAxes))
            cells = (
                str(candidate.get("rank", "")),
                str(candidate.get("selection_label") or ""),
                display,
                f"{float(candidate.get('score') or 0):.4f}",
            )
            x = x0
            for cell_index, (cell, width) in enumerate(zip(cells, widths)):
                if cell_index == 2:
                    _draw_wrapped_text(ax, x + 0.008, y + row_h - 0.018, cell, max_units=18, line_gap=0.014, color="#26323b", fontsize=7.8, transform=ax.transAxes)
                else:
                    ax.text(x + 0.008, y + row_h - 0.018, cell, color="#26323b", fontsize=7.8, transform=ax.transAxes)
                x += width
            reason_x = x0 + sum(widths[:-1]) + 0.008
            _draw_wrapped_text(ax, reason_x, y + row_h - 0.018, reason, max_units=50, line_gap=0.014, color="#26323b", fontsize=7.2, transform=ax.transAxes)
            y -= row_h
        missing = row.get("missing_price_tickers") or ""
        source = row.get("source_summary") or ""
        if missing or source:
            note = "；".join(part for part in [f"來源：{source}" if source else "", f"缺價：{missing}" if missing else ""] if part)
            note_lines = _wrap_text_lines(note, max_units=78)
            _draw_wrapped_text(ax, x0 + 0.008, y - 0.004, note, max_units=78, line_gap=0.012, color="#66737d", fontsize=7.1, transform=ax.transAxes)
            y -= 0.012 * max(len(note_lines), 1) + 0.006
        y -= 0.012
    for row in skipped_rows[:3]:
        if y < 0.09:
            break
        ax.text(x0, y, f"暫無正式觀察：{row['pool_name']}，原因：{_sanitize_visible_report_reason(row['reason'])}", color="#b42318", fontsize=8.0, transform=ax.transAxes)
        y -= 0.024
    return max(y, 0.095)


def _top_candidate_rows(observation: StockPoolObservation, limit: int = 3) -> list[dict[str, Any]]:
    strength_order = sorted(
        observation.candidates,
        key=lambda item: (item.score, item.ret20, item.ticker),
        reverse=True,
    )
    strength_rank_by_ticker = {candidate.ticker: rank for rank, candidate in enumerate(strength_order, start=1)}
    candidates = sorted(
        observation.candidates,
        key=lambda item: (item.passed, item.score, item.ret20, item.ticker),
        reverse=True,
    )
    rows = []
    asset_type_by_ticker = (observation.source_metadata or {}).get("candidate_asset_types") or {}
    gate_rule_id = str((observation.source_metadata or {}).get("gate_rule_id") or observation.gate_rule_id or "")
    gate_details_by_ticker = (observation.source_metadata or {}).get("candidate_gate_details") or {}
    for rank, candidate in enumerate(candidates[:limit], start=1):
        asset_type = _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker)
        gate = gate_details_by_ticker.get(candidate.ticker) or _candidate_gate_evaluation(
            candidate,
            asset_type,
            gate_rule_id=gate_rule_id or "universal_pool_base_gate_v1",
            attack_gate_active=(observation.source_metadata or {}).get("attack_gate_active"),
        )
        selection_layer = str(gate["selection_layer"])
        rows.append(
            {
                "rank": rank,
                "strength_rank": strength_rank_by_ticker.get(candidate.ticker, rank),
                "ticker": candidate.ticker,
                "display": _candidate_display(observation, candidate.ticker),
                "asset_type": asset_type,
                "score": round(candidate.score, 6),
                "rank_score": gate["rank_score"],
                "passed": candidate.passed,
                "base_pool_passed": gate["base_pool_passed"],
                "attack_gate_open": gate["attack_gate_open"],
                "eligible_for_pool_selection": gate["eligible_for_pool_selection"],
                "selection_layer": selection_layer,
                "selection_label": _selection_label(selection_layer),
                "gate_rule_id": gate["gate_rule_id"],
                "gate_reason": gate["gate_reason"],
                "is_model_target": candidate.ticker == observation.top_ticker,
                "flow_risk_score": round(candidate.flow_risk_score, 6),
                "institutional_risk": round(candidate.institutional_risk, 6),
                "bullish_flow_score": round(candidate.bullish_flow_score, 6),
                "flow_source_dates": candidate.flow_source_dates,
                "flow_source_kinds": candidate.flow_source_kinds,
                "reason": _candidate_reason(observation, candidate),
            }
        )
    return rows


def _candidate_display(observation: StockPoolObservation, ticker: str) -> str:
    if ticker == observation.top_ticker and observation.top_display:
        return _normalize_display_label(observation.top_display, ticker)
    symbol = ticker.replace(".TW", "").replace(".TWO", "")
    metadata = observation.source_metadata or {}
    for candidate_symbol, candidate_display in zip(
        metadata.get("candidate_symbols") or [],
        metadata.get("candidate_displays") or [],
    ):
        if str(candidate_symbol) == symbol and candidate_display:
            return _normalize_display_label(str(candidate_display), ticker)
    known = KNOWN_SYMBOLS.get(ticker, {})
    symbol = known.get("symbol") or symbol
    name = known.get("name") or symbol
    return f"{name}({symbol})"


def _candidate_reason(observation: StockPoolObservation, candidate: UniversalCandidateScore) -> str:
    if observation.strategy_preset in {"best_v20260605", "ai_theme_large_cap_v20260613"}:
        metadata = observation.source_metadata or {}
        regime = metadata.get("market_regime_label") or "目前市場環境"
        gate_text = "個股攻擊閘門已開啟" if metadata.get("attack_gate_active") else "個股攻擊閘門未開啟"
        if candidate.ticker == observation.top_ticker:
            return f"最終模型目標；{regime}，{gate_text}，先採市場曝險工具。"
        return f"強弱分數靠前，但{gate_text}，尚未切回個股攻擊；{candidate.reason or '觀察'}。"
    parts = []
    if candidate.passed:
        parts.append("通過池內條件")
    else:
        parts.append(candidate.reason or "未通過池內條件")
    if candidate.ret60:
        parts.append(f"60日動能{candidate.ret60:+.1%}")
    if candidate.flow_risk_score > 0:
        parts.append(f"風險分{candidate.flow_risk_score:.2f}")
    if candidate.flow_risk_reasons:
        parts.append(candidate.flow_risk_reasons)
    return "；".join(parts)[:80]


def _top_candidates_text(candidates: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{row.get('rank')}.{row.get('display')}[{row.get('selection_label') or _selection_label(str(row.get('selection_layer') or ''))}]({float(row.get('score') or 0):.2f})"
        for row in candidates[:3]
    )


def _selection_label(selection_layer: str) -> str:
    if selection_layer == SELECTION_FORMAL_CANDIDATE:
        return "正式候選"
    if selection_layer == SELECTION_MARKET_EXPOSURE_TOOL:
        return "曝險工具"
    if selection_layer == SELECTION_OBSERVATION_ONLY:
        return "觀察"
    return "從缺"


def _candidate_review_label(value: object) -> str:
    key = str(value or "").strip().lower()
    if key == "monthly":
        return "月頻"
    if key == "weekly":
        return "週頻"
    if key == "daily":
        return "日頻"
    return str(value or "").strip() or "-"


def _short_pool_name(item: dict[str, Any]) -> str:
    pool_id = str(item.get("pool_id") or "")
    if pool_id in POOL_SHORT_NAMES:
        return POOL_SHORT_NAMES[pool_id]
    name = str(item.get("pool_name") or item.get("name") or pool_id or "股票池")
    replacements = [
        ("AI中大型權值股池最佳版", "AI主線池"),
        ("動態0050成分股池", "大型廣度池"),
        ("大型核心權值股池", "風格補強池"),
        ("核心防守風格池", "風格補強池"),
        ("核心風格補強池", "風格補強池"),
        ("雷達中小型校準版", "雷達池"),
    ]
    for old, new in replacements:
        if old in name:
            return new
    return name[:8]


def _vote_color(row: dict[str, Any], consensus: dict[str, Any], index: int) -> str:
    state = consensus.get("result_state")
    winner = consensus.get("winner_ticker")
    if state == "consensus":
        return VOTE_WINNER_COLOR if row.get("top_ticker") == winner else VOTE_MINOR_COLOR
    if state == "divergent":
        return VOTE_DIVERGENT_COLORS[index % len(VOTE_DIVERGENT_COLORS)]
    return VOTE_NEUTRAL_COLOR


def _compact_display(value: str, *, limit: int = 16) -> str:
    text = str(value).replace("（", "(").replace("）", ")").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_twd(value: object) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):,.0f} 元"
    except (TypeError, ValueError):
        return str(value)


def _display_width_units(value: str) -> int:
    width = 0
    for char in str(value):
        width += 1 if ord(char) < 128 else 2
    return width


def _wrap_text_lines(value: str, *, max_units: int) -> list[str]:
    text = str(value or "").replace("（", "(").replace("）", ")").strip()
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and _display_width_units(candidate) > max_units:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _draw_wrapped_text(ax, x: float, y: float, value: str, *, max_units: int, line_gap: float, **kwargs) -> list[str]:
    lines = _wrap_text_lines(value, max_units=max_units)
    for index, line in enumerate(lines):
        ax.text(x, y - index * line_gap, line, **kwargs)
    return lines


def _normalize_display_label(value: str, ticker: str) -> str:
    text = str(value or "").replace("（", "(").replace("）", ")").strip()
    symbol = str(ticker or "").replace(".TW", "").replace(".TWO", "")
    known = KNOWN_SYMBOLS.get(ticker, {})
    name = known.get("name")
    if name and symbol:
        return f"{name}({symbol})"
    if text and "(" in text and ")" in text:
        return text
    if text and symbol and symbol not in text:
        return f"{text}({symbol})"
    if text:
        return text
    return ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a unified stock-pool observation snapshot.")
    parser.add_argument("--pool-store", default="work/stock_pools/stock_pools.json")
    parser.add_argument("--pool-id", default="large_cap_best_v20260605")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--cache-dir", default="backtest_cache/stock_pool_observations")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--warmup-start", default="2020-01-02")
    parser.add_argument("--radar-snapshot-dir", default=os.getenv("RADAR_SNAPSHOT_DIR", ""))
    parser.add_argument("--radar-data-dir", default=os.getenv("RADAR_DATA_DIR", ""))
    parser.add_argument("--market-cap-data", default=os.getenv("MARKET_CAP_DATA_PATH", ""))
    parser.add_argument("--institutional-flow-data", default=os.getenv("INSTITUTIONAL_FLOW_DATA_PATH", ""))
    parser.add_argument("--margin-short-data", default=os.getenv("MARGIN_SHORT_DATA_PATH", ""))
    parser.add_argument("--borrow-lending-data", default=os.getenv("BORROW_LENDING_DATA_PATH", ""))
    parser.add_argument("--day-trading-data", default=os.getenv("DAY_TRADING_DATA_PATH", ""))
    parser.add_argument("--sentiment-data", default=os.getenv("SENTIMENT_DATA_PATH", ""))
    parser.add_argument("--valuation-data", default=os.getenv("VALUATION_DATA_PATH", ""))
    parser.add_argument("--tw50-constituents", default=os.getenv("TW50_CONSTITUENTS_PATH", ""))
    parser.add_argument("--radar-top-n", type=int, default=20)
    parser.add_argument("--require-exact-signal-date", action="store_true")
    parser.add_argument("--require-fresh-institutional-flow", action="store_true")
    parser.add_argument(
        "--include-non-operational-pools",
        action="store_true",
        help="Also include task/scorecard pools that are not meant for the operational observation PDF.",
    )
    args = parser.parse_args()

    store = StockPoolStore(args.pool_store)
    pools = store.list_pools()
    if args.pool_id == "all":
        manifest = run_stock_pool_observation_batch(
            pools=pools,
            signal_date=args.signal_date,
            warmup_start=args.warmup_start,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            radar_snapshot_dir=args.radar_snapshot_dir or None,
            radar_data_dir=args.radar_data_dir or None,
            market_cap_data=args.market_cap_data or None,
            institutional_flow_data=args.institutional_flow_data or None,
            margin_short_data=args.margin_short_data or None,
            borrow_lending_data=args.borrow_lending_data or None,
            day_trading_data=args.day_trading_data or None,
            sentiment_data=args.sentiment_data or None,
            valuation_data=args.valuation_data or None,
            tw50_constituents_path=args.tw50_constituents or None,
            radar_top_n=args.radar_top_n,
            require_exact_signal_date=args.require_exact_signal_date,
            require_fresh_institutional_flow=args.require_fresh_institutional_flow,
            operational_only=not args.include_non_operational_pools,
        )
        print(f"STOCK_POOL_OBSERVATION_MANIFEST={Path(manifest['output_root']).resolve() / 'stock_pool_observation_manifest.json'}")
        return
    pool = next((item for item in pools if item["pool_id"] == args.pool_id), None)
    if pool is None:
        raise ValueError(f"Unknown pool_id: {args.pool_id}")
    pool = _resolve_dynamic_observation_pool(
        pool,
        signal_date=args.signal_date,
        radar_snapshot_dir=args.radar_snapshot_dir or None,
        radar_data_dir=args.radar_data_dir or None,
        radar_top_n=args.radar_top_n,
        tw50_constituents_path=args.tw50_constituents or None,
    )
    tickers = [symbol["ticker"] for symbol in pool.get("resolved_symbols", [])]
    if not tickers:
        raise ValueError(f"Pool has no resolved tickers: {args.pool_id}")
    prices, missing_price_tickers = _load_observation_price_frames(
        tickers=tickers,
        start_date=_price_start_for_pool(pool, args.warmup_start),
        end_date=args.signal_date,
        cache_dir=args.cache_dir,
    )
    if not prices:
        raise ValueError(f"No price data available for pool tickers: {', '.join(tickers)}")
    market_caps, market_cap_source = load_first_available_market_caps(
        signal_date=args.signal_date,
        explicit_path=args.market_cap_data or None,
        radar_data_dir=args.radar_data_dir or None,
    )
    if market_cap_source:
        print(f"STOCK_POOL_OBSERVATION_MARKET_CAP_SOURCE={Path(market_cap_source).resolve()}")
    risk_signals, risk_sources = load_first_available_risk_factors(
        signal_date=args.signal_date,
        radar_data_dir=args.radar_data_dir or None,
        institutional_path=args.institutional_flow_data or None,
        margin_short_path=args.margin_short_data or None,
        borrow_lending_path=args.borrow_lending_data or None,
        day_trading_path=args.day_trading_data or None,
        sentiment_path=args.sentiment_data or None,
    )
    if risk_sources:
        print(f"STOCK_POOL_OBSERVATION_RISK_FACTOR_SOURCES={json.dumps(risk_sources, ensure_ascii=False)}")
    valuation_signals = load_valuation_signals(
        args.valuation_data or None,
        signal_date=args.signal_date,
        current_price_by_ticker=_current_close_by_ticker(prices, args.signal_date),
    )
    if valuation_signals:
        print(f"STOCK_POOL_OBSERVATION_VALUATION_SOURCE={Path(args.valuation_data).resolve()}")
    observation = build_dispatched_stock_pool_observation(
        pool=pool,
        prices_by_ticker=prices,
        signal_date=args.signal_date,
        warmup_start=args.warmup_start,
        market_cap_by_ticker=market_caps,
        risk_signal_by_ticker=risk_signals,
        valuation_signal_by_ticker=valuation_signals,
        require_exact_signal_date=args.require_exact_signal_date,
    )
    if missing_price_tickers:
        print(f"STOCK_POOL_OBSERVATION_MISSING_PRICE_TICKERS={','.join(missing_price_tickers)}")
    output_dir = Path(args.output_root) / args.pool_id / args.signal_date.replace("-", "")
    write_stock_pool_observation(output_dir, observation)
    print(f"STOCK_POOL_OBSERVATION_DIR={output_dir.resolve()}")


def _resolve_signal_date(prices_by_ticker: dict[str, pd.DataFrame], requested: pd.Timestamp) -> pd.Timestamp:
    common = None
    for frame in prices_by_ticker.values():
        dates = set(frame.index[frame.index <= requested])
        common = dates if common is None else common & dates
    if not common:
        raise ValueError(f"No common signal date on or before {requested.strftime('%Y-%m-%d')}")
    return max(common)


def _current_close_by_ticker(prices_by_ticker: dict[str, pd.DataFrame], signal_date: str | pd.Timestamp) -> dict[str, float]:
    signal_ts = pd.Timestamp(signal_date).normalize()
    closes: dict[str, float] = {}
    for ticker, frame in prices_by_ticker.items():
        history = frame.loc[frame.index <= signal_ts]
        if history.empty:
            continue
        column = "adj_close" if "adj_close" in history.columns else "close"
        close = pd.to_numeric(history[column], errors="coerce").dropna()
        if close.empty:
            continue
        closes[ticker] = float(close.iloc[-1])
    return closes


def _build_regime_signal_observation(
    *,
    pool: dict[str, Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    warmup_start: str,
    market_cap_by_ticker: dict[str, float] | None,
    risk_signal_by_ticker: dict[str, RiskFactorSignal] | None,
    valuation_signal_by_ticker: dict[str, ValuationSignal] | None,
    require_exact_signal_date: bool,
    variant: RegimeModeSwitchVariant,
    strategy_id: str,
) -> StockPoolObservation:
    requested_ts = pd.Timestamp(signal_date)
    signal_ts = _resolve_signal_date(prices_by_ticker, requested_ts)
    if require_exact_signal_date and signal_ts != requested_ts.normalize():
        raise ValueError(
            f"No exact common price data for signal date {requested_ts.strftime('%Y-%m-%d')}; "
            f"latest common date is {signal_ts.strftime('%Y-%m-%d')}"
        )

    config = load_config("configs/ep05_universe.json")
    group = config.group_by_id(FROZEN_BEST_GROUP_ID)
    labels = {asset.ticker: asset.label for asset in group.assets}
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    required_tickers = set(labels)
    missing = sorted(required_tickers - set(prices_by_ticker))
    if missing:
        raise ValueError(f"best_v20260605 requires full large-cap pool prices: {', '.join(missing)}")

    signal = build_frozen_strategy_signal(
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        replay_start=warmup_start,
        prices_by_ticker={ticker: prices_by_ticker[ticker] for ticker in sorted(required_tickers)},
        labels=labels,
        asset_types=asset_types,
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
        manual_splits=config.manual_splits,
        variant=variant,
        strategy_id=strategy_id,
    )
    profile = infer_pool_profile(prices_by_ticker, signal_ts)
    params = default_parameters_for_profile(profile)
    universal = score_universal_candidates(
        prices_by_ticker,
        signal_ts,
        params,
        market_cap_by_ticker=market_cap_by_ticker,
        risk_signal_by_ticker=risk_signal_by_ticker,
        valuation_signal_by_ticker=valuation_signal_by_ticker,
    )
    candidates: list[UniversalCandidateScore] = []
    for row in signal.ranking:
        ticker = str(row["ticker"])
        base = universal.get(ticker) or UniversalCandidateScore(
            ticker=ticker,
            score=0.0,
            ret20=0.0,
            ret60=0.0,
            ret120=0.0,
            vol20=0.0,
            avg_turnover_twd=0.0,
            drawdown20=0.0,
            passed=False,
            reason="no_price_score",
        )
        passed = signal.target_is_actionable and ticker == signal.target_ticker
        candidates.append(
            UniversalCandidateScore(
                ticker=ticker,
                score=float(row["score"]),
                ret20=base.ret20,
                ret60=base.ret60,
                ret120=base.ret120,
                vol20=base.vol20,
                avg_turnover_twd=base.avg_turnover_twd,
                drawdown20=base.drawdown20,
                passed=passed,
                reason=signal.model_target_status if passed else str(row.get("score_band") or ""),
                liquidity_profile=base.liquidity_profile,
                size_profile=base.size_profile,
                market_cap_twd=base.market_cap_twd,
                size_basis=base.size_basis,
                profile_type=base.profile_type,
                applied_score_mode=base.applied_score_mode,
                flow_risk_score=base.flow_risk_score,
                institutional_risk=base.institutional_risk,
                margin_risk=base.margin_risk,
                borrow_risk=base.borrow_risk,
                day_trading_risk=base.day_trading_risk,
                sentiment_risk=base.sentiment_risk,
                bullish_flow_score=base.bullish_flow_score,
                sentiment_score=base.sentiment_score,
                flow_score_adjustment=base.flow_score_adjustment,
                flow_risk_reasons=base.flow_risk_reasons,
                flow_source_dates=base.flow_source_dates,
                flow_source_kinds=base.flow_source_kinds,
                valuation_score_adjustment=base.valuation_score_adjustment,
                valuation_gate_passed=base.valuation_gate_passed,
                valuation_safety_margin_pct=base.valuation_safety_margin_pct,
                valuation_fair_price=base.valuation_fair_price,
                valuation_buy_price=base.valuation_buy_price,
                valuation_reason=base.valuation_reason,
                valuation_source_date=base.valuation_source_date,
            )
        )
    candidates = [
        candidate
        for candidate in candidates
        if not _exclude_from_formal_candidate_universe(pool, candidate.ticker)
    ]
    top_ticker = (
        signal.target_ticker
        if signal.target_is_actionable and not _exclude_from_formal_candidate_universe(pool, signal.target_ticker)
        else None
    )
    display_by_ticker = {
        symbol["ticker"]: _normalize_display_label(
            symbol.get("display") or labels.get(symbol["ticker"], symbol["ticker"]),
            symbol["ticker"],
        )
        for symbol in pool.get("resolved_symbols", [])
    }
    top_asset_type = asset_types.get(top_ticker) if top_ticker else None
    top_candidate = next((candidate for candidate in candidates if candidate.ticker == top_ticker), None)
    gate_rule_id = _gate_rule_id_for_pool(pool)
    top_gate = _candidate_gate_evaluation(
        top_candidate,
        top_asset_type,
        gate_rule_id=gate_rule_id,
        attack_gate_active=signal.attack_gate_active,
    )
    return StockPoolObservation(
        schema_version=1,
        pool_id=str(pool["pool_id"]),
        pool_name=str(pool["name"]),
        strategy_preset=str(pool.get("strategy_preset") or "best_v20260605"),
        signal_date=signal.signal_date,
        data_end_date=signal.signal_date,
        candidate_count=len(candidates),
        passed_count=sum(1 for candidate in candidates if candidate.passed),
        pool_profile=profile,
        parameters=params,
        top_ticker=top_ticker,
        top_display=display_by_ticker.get(top_ticker, labels.get(top_ticker, top_ticker)) if top_ticker else None,
        top_score=next((candidate.score for candidate in candidates if candidate.ticker == top_ticker), None),
        rank_score=top_gate["rank_score"],
        base_pool_passed=top_gate["base_pool_passed"],
        gate_rule_id=top_gate["gate_rule_id"],
        gate_reason=top_gate["gate_reason"],
        action_state=signal.model_target_status,
        candidates=candidates,
        top_asset_type=top_asset_type,
        attack_gate_open=top_gate["attack_gate_open"] if top_ticker else None,
        eligible_for_pool_selection=top_gate["eligible_for_pool_selection"],
        selection_layer=top_gate["selection_layer"] if top_ticker else SELECTION_NO_SELECTION,
        selection_reason=top_gate["gate_reason"] if top_ticker else "正式模型目前沒有可投票的池內目標。",
        source_metadata={
            "source_type": f"{strategy_id}_signal",
            "market_regime_label": signal.market_regime_label,
            "attack_gate_active": signal.attack_gate_active,
            "attack_gate_ever_activated": signal.attack_gate_ever_activated,
            "risk_off_active": signal.risk_off_active,
            "action": signal.action,
            "target_label": signal.target_label,
            "target_exposure": signal.target_exposure,
            "candidate_asset_types": asset_types,
            "gate_rule_id": gate_rule_id,
        },
        decision_layer=FORMAL_TRADE_SIGNAL,
        active_in_trade_decision=True,
        source_module="frozen_strategy_monitor",
    )


def _observation_pools(pools: list[dict[str, Any]], *, operational_only: bool) -> list[dict[str, Any]]:
    if not operational_only:
        return pools
    return [
        pool
        for pool in pools
        if pool.get("operational_observation", True)
    ]


def _load_observation_price_frames(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        try:
            loaded = download_yfinance_prices(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                allow_edge_gap=False,
            )
            prices.update(loaded)
        except Exception:
            try:
                lagged = download_yfinance_prices(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date,
                    cache_dir=cache_dir,
                    allow_edge_gap=True,
                )
            except Exception:
                lagged = {}
            if lagged:
                prices.update(lagged)
                continue
            cached = _load_current_cached_price_frame(
                ticker=ticker,
                end_date=end_date,
                cache_dir=cache_dir,
            )
            if cached is not None:
                prices[ticker] = cached
            else:
                missing.append(ticker)
    incomplete = incomplete_tickers(prices, end_date)
    if incomplete:
        prices = fill_signal_date_from_twse(prices, end_date, incomplete)
        write_price_cache(Path(cache_dir), prices, incomplete)
        still_incomplete = set(incomplete_tickers(prices, end_date))
        missing = sorted(set(missing).union(still_incomplete))
    return prices, missing


def _load_current_cached_price_frame(
    *,
    ticker: str,
    end_date: str,
    cache_dir: str | Path,
    min_rows: int = 260,
) -> pd.DataFrame | None:
    csv_path = Path(cache_dir) / f"{ticker.replace('.', '_')}.csv"
    if not csv_path.exists():
        return None
    try:
        frame = load_price_csv(csv_path)
    except Exception:
        return None
    if frame.empty or len(frame) < min_rows:
        return None
    latest = frame.index.max()
    if latest < pd.Timestamp(end_date).normalize():
        return None
    return frame


def _price_start_for_pool(pool: dict[str, Any], warmup_start: str) -> str:
    if pool.get("strategy_preset") in {"best_v20260605", "ai_theme_large_cap_v20260613"}:
        return (pd.Timestamp(warmup_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    return warmup_start


def _observation_price_tickers(pool: dict[str, Any], candidate_tickers: list[str]) -> list[str]:
    tickers = list(candidate_tickers)
    if str(pool.get("pool_id") or "") == "tw50_dynamic_constituents_v0" and TW50_ATTACK_GATE_BENCHMARK not in tickers:
        tickers.append(TW50_ATTACK_GATE_BENCHMARK)
    if str(pool.get("strategy_preset") or "") == "core_defensive_style_v1" and CORE_DEFENSIVE_BENCHMARK not in tickers:
        tickers.append(CORE_DEFENSIVE_BENCHMARK)
    return tickers


def _resolve_dynamic_observation_pool(
    pool: dict[str, Any],
    *,
    signal_date: str,
    radar_snapshot_dir: str | Path | None,
    radar_data_dir: str | Path | None,
    radar_top_n: int,
    tw50_constituents_path: str | Path | None,
) -> dict[str, Any]:
    if str(pool.get("strategy_preset") or "") == "core_defensive_style_v1":
        updated = _resolve_core_defensive_style_representatives(pool, signal_date=signal_date)
        if updated is not None:
            return updated
    dynamic = pool.get("dynamic_constituents") or {}
    if not pool.get("resolved_symbols") and dynamic.get("source") == "tw50_history_csv":
        path = tw50_constituents_path or dynamic.get("path")
        if not path:
            return pool
        updated = json.loads(json.dumps(pool, ensure_ascii=False))
        try:
            updated["resolved_symbols"] = load_tw50_constituents_for_date(path, signal_date)
        except (FileNotFoundError, ValueError):
            return pool
        updated["tw50_constituent_source"] = str(path)
        return updated
    if pool.get("resolved_symbols") or pool.get("strategy_preset") != "radar_core_mid_small_calibrated_v1":
        return pool
    data_dir = _resolve_radar_data_dir(radar_data_dir=radar_data_dir, radar_snapshot_dir=radar_snapshot_dir)
    if data_dir is None:
        return pool
    try:
        candidates = load_formal_radar_candidates(data_dir, signal_date=signal_date)
    except FileNotFoundError:
        return pool
    updated = json.loads(json.dumps(pool, ensure_ascii=False))
    updated["resolved_symbols"] = formal_radar_candidates_to_symbols(candidates[:radar_top_n])
    updated["radar_candidate_source"] = str(data_dir)
    updated["radar_candidate_mode"] = "formal_bucket_actionable_else_watch"
    return updated


def _resolve_core_defensive_style_representatives(
    pool: dict[str, Any],
    *,
    signal_date: str,
) -> dict[str, Any] | None:
    config = pool.get("candidate_review_config") or {}
    if str(config.get("source_mode") or "") != "core_defensive_candidate_csv":
        return None
    source_path = str(config.get("path") or "").strip()
    if not source_path:
        return None
    path = Path(source_path)
    if not path.exists():
        return None
    source_candidates = load_core_defensive_candidate_source(path, signal_date=signal_date)
    representatives = _select_core_defensive_style_representatives(source_candidates)
    if not representatives:
        return None
    updated = json.loads(json.dumps(pool, ensure_ascii=False))
    updated["resolved_symbols"] = [
        _core_defensive_candidate_to_symbol(item)
        for item in representatives
    ]
    updated["core_defensive_style_source"] = str(path)
    updated["core_defensive_style_selection_mode"] = "one_representative_per_style_bucket_v1"
    updated["core_defensive_source_candidate_count"] = len(source_candidates)
    updated["core_defensive_representative_count"] = len(representatives)
    updated["core_defensive_style_buckets"] = [
        {
            "style_bucket": item["style_bucket"],
            "ticker": item["ticker"],
            "display": item.get("display", item["ticker"]),
            "role": item.get("role", ""),
        }
        for item in representatives
    ]
    return updated


def _select_core_defensive_style_representatives(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    market_exposure: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("review_status") != "active" or not item.get("is_current_member"):
            continue
        ticker = str(item.get("ticker") or "").strip()
        if not ticker:
            continue
        asset_type = _normalize_asset_type(None, ticker)
        if asset_type == ASSET_TYPE_STOCK and ticker in CORE_STYLE_COMPLEMENT_EXCLUDED_TICKERS:
            continue
        payload = dict(item)
        payload["asset_type"] = asset_type if asset_type != ASSET_TYPE_UNKNOWN else ASSET_TYPE_STOCK
        payload["style_bucket"] = _core_defensive_style_bucket(item)
        if payload["style_bucket"] == CORE_DEFENSIVE_MARKET_EXPOSURE_BUCKET:
            market_exposure.append(payload)
            continue
        current = selected.get(payload["style_bucket"])
        if current is None or _core_defensive_representative_sort_key(payload) > _core_defensive_representative_sort_key(current):
            selected[payload["style_bucket"]] = payload
    market_exposure.sort(key=_core_defensive_representative_sort_key, reverse=True)
    stock_representatives = sorted(
        selected.values(),
        key=lambda item: (str(item.get("style_bucket") or ""), item.get("ticker", "")),
    )
    return [*market_exposure, *stock_representatives]


def _core_defensive_style_bucket(item: dict[str, Any]) -> str:
    ticker = str(item.get("ticker") or "").strip()
    if _normalize_asset_type(None, ticker) == ASSET_TYPE_ETF:
        return CORE_DEFENSIVE_MARKET_EXPOSURE_BUCKET
    role = str(item.get("role") or "").strip()
    if "半導體" in role:
        return "semiconductor_core"
    if "電信" in role:
        return "telecom_defensive"
    if "金融" in role:
        return "financial_core"
    if "消費耐久" in role:
        return "consumer_durable"
    if "消費" in role or "通路" in role:
        return "consumer_defensive"
    if "航運" in role or "景氣循環" in role:
        return "cyclical_core"
    if "傳產" in role or "塑化" in role or "基建" in role:
        return "traditional_materials"
    if "非AI科技" in role or "工業電腦" in role:
        return "non_ai_technology"
    return role or ticker


def _core_defensive_representative_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        _number(item.get("defensive_score")),
        _number(item.get("stability_score")),
        _number(item.get("cross_sector_score")),
        _number(item.get("fundamental_score")),
        str(item.get("ticker") or ""),
    )


def _core_defensive_candidate_to_symbol(item: dict[str, Any]) -> dict[str, Any]:
    ticker = str(item.get("ticker") or "").strip()
    known = KNOWN_SYMBOLS.get(ticker, {})
    symbol = str(known.get("symbol") or ticker.split(".")[0]).strip()
    display = str(item.get("display") or known.get("name") or ticker).strip()
    return {
        "ticker": ticker,
        "symbol": symbol,
        "name": str(known.get("name") or display.split("(")[0] or symbol).strip(),
        "display": display,
        "source": "core_style_complement_representative",
        "asset_type": item.get("asset_type") or known.get("asset_type") or ASSET_TYPE_STOCK,
        "style_bucket": item.get("style_bucket", ""),
        "style_role": item.get("role", ""),
        "review_status": item.get("review_status", ""),
        "is_current_member": bool(item.get("is_current_member")),
    }


def _resolve_radar_data_dir(
    *,
    radar_data_dir: str | Path | None,
    radar_snapshot_dir: str | Path | None,
) -> Path | None:
    if radar_data_dir:
        return Path(radar_data_dir)
    if not radar_snapshot_dir:
        return None
    snapshot_path = Path(radar_snapshot_dir)
    if snapshot_path.name == "history":
        return snapshot_path.parent
    return snapshot_path


if __name__ == "__main__":
    main()
