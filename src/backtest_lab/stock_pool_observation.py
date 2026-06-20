from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.config import load_config
from backtest_lab.data import download_yfinance_prices
from backtest_lab.decision_layers import (
    CANDIDATE_SOURCE,
    FORMAL_TRADE_SIGNAL,
    DATA_READINESS,
    default_stock_pool_model_layer_audit,
    write_model_layer_audit,
)
from backtest_lab.formal_radar_candidates import formal_radar_candidates_to_symbols, load_formal_radar_candidates
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
from backtest_lab.stock_pool_candidate_review import build_candidate_review, write_candidate_reviews
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
)
from backtest_lab.valuation_source import ValuationSignal, load_valuation_signals


DEFAULT_OUTPUT_ROOT = "outputs/stock_pool_observations"
REPORT_NAME = "AI股票池觀察總覽"
REPORT_TITLE = "AI股票池三池表決觀察總覽"
REPORT_VERSION = "v20260612"
REPORT_LATEST_FILENAME = f"{REPORT_NAME}_最新版_{REPORT_VERSION}.pdf"
FROZEN_BEST_GROUP_ID = "group_c_0050_00631l_plus_mega_caps"
POOL_SHORT_NAMES = {
    "ai_theme_large_cap_v20260613": "AI主線池",
    "tw50_dynamic_constituents_v0": "大型廣度池",
    "large_core_bluechip_v0": "核心防守池",
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
    available_symbols = [
        symbol
        for symbol in pool.get("resolved_symbols") or pool.get("symbols") or []
        if symbol.get("ticker") in prices_by_ticker
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
        symbol["ticker"]: symbol.get("display") or symbol["ticker"]
        for symbol in available_symbols
    }
    asset_type_by_ticker = _asset_type_by_ticker(available_symbols)
    top = _first_eligible_candidate(candidates, asset_type_by_ticker=asset_type_by_ticker)
    top_asset_type = _asset_type_for_ticker(top.ticker, asset_type_by_ticker) if top else None
    source_metadata = _build_pool_source_metadata(pool, available_symbols)
    source_metadata["candidate_asset_types"] = asset_type_by_ticker
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
        action_state="watch_candidate" if top else "no_valid_candidate",
        candidates=candidates,
        top_asset_type=top_asset_type,
        attack_gate_open=_attack_gate_open_for_candidate(top, top_asset_type) if top else None,
        eligible_for_pool_selection=bool(top),
        selection_layer=_selection_layer_for_candidate(top, top_asset_type) if top else SELECTION_NO_SELECTION,
        selection_reason=_selection_reason_for_candidate(top, top_asset_type) if top else "池內沒有通過入選條件的正式候選或市場曝險工具。",
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
) -> UniversalCandidateScore | None:
    return next(
        (
            candidate
            for candidate in candidates
            if _eligible_for_pool_selection(candidate, _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker))
        ),
        None,
    )


def _eligible_for_pool_selection(candidate: UniversalCandidateScore | None, asset_type: str | None) -> bool:
    if candidate is None:
        return False
    layer = _selection_layer_for_candidate(candidate, asset_type)
    return layer in {SELECTION_FORMAL_CANDIDATE, SELECTION_MARKET_EXPOSURE_TOOL}


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
        return f"動態0050成分股：{metadata.get('candidate_count', 0)}檔，來源 {metadata.get('source_path') or '未標'}"
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
        "signal_date": signal_date,
        "require_exact_signal_date": require_exact_signal_date,
        "operational_only": operational_only,
        "market_cap_source": market_cap_source,
        "market_cap_count": len(market_caps),
        "risk_factor_sources": risk_sources,
        "risk_factor_count": len(risk_signals),
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
            prices, missing_price_tickers = _load_observation_price_frames(
                tickers=tickers,
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
                    "rank": 1 if observation.top_ticker else None,
                    "attack_gate_open": observation.attack_gate_open,
                    "eligible_for_pool_selection": observation.eligible_for_pool_selection,
                    "selection_layer": observation.selection_layer,
                    "selection_reason": observation.selection_reason,
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
    manifest["consensus"] = build_consensus(manifest)
    manifest["model_layer_audit"] = default_stock_pool_model_layer_audit(
        signal_date=signal_date,
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
                "rank": item.get("rank", ""),
                "attack_gate_open": item.get("attack_gate_open", ""),
                "eligible_for_pool_selection": item.get("eligible_for_pool_selection", False),
                "selection_layer": item.get("selection_layer", ""),
                "selection_reason": item.get("selection_reason", ""),
                "action_state": item.get("action_state", ""),
                "decision_layer": item.get("decision_layer", ""),
                "active_in_trade_decision": item.get("active_in_trade_decision", False),
                "source_module": item.get("source_module", ""),
                "report_line": (item.get("dispatch") or {}).get("report_line", ""),
                "workflow_file": (item.get("dispatch") or {}).get("workflow_file", ""),
                "missing_price_tickers": ",".join(item.get("missing_price_tickers") or []),
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
                "rank": "",
                "attack_gate_open": "",
                "eligible_for_pool_selection": False,
                "selection_layer": SELECTION_NO_SELECTION,
                "selection_reason": item.get("reason", ""),
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
                "reason": item.get("reason", ""),
                "output_dir": "",
            }
        )
    pd.DataFrame(rows).to_csv(root / "stock_pool_observation_summary.csv", index=False, encoding="utf-8-sig")
    (root / "stock_pool_observation_report.md").write_text(
        markdown_observation_batch_report(manifest, rows),
        encoding="utf-8",
    )
    if manifest.get("generated"):
        write_stock_pool_observation_batch_pdf(root / REPORT_LATEST_FILENAME, manifest, rows)


def markdown_observation_batch_report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    consensus = manifest.get("consensus") or {}
    lines = [
        "# 股票池觀察摘要",
        "",
        f"- 訊號日：{manifest.get('signal_date', '')}",
        f"- 已產出股票池：{len(manifest.get('generated', []))}",
        f"- 跳過股票池：{len(manifest.get('skipped', []))}",
        f"- 三池共識：{consensus.get('winner_display') or '沒有形成明確共識'}",
        f"- 表決原因：{consensus.get('reason') or '尚未產生三池表決結果'}",
        "",
        "| 狀態 | 股票池 | 角色 | 成員檢查 | 前三名 / 跳過原因 | 來源摘要 | 缺價股票 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["status"] == "generated":
            target = row["top_candidates_text"] or row["top_display"] or row["top_ticker"] or row["action_state"] or "無合格候選"
            missing = row["missing_price_tickers"] or "-"
        else:
            target = row["reason"] or "skipped"
            missing = "-"
        role = row.get("role_name") or "-"
        review_frequency = _candidate_review_label(row.get("candidate_review_frequency"))
        lines.append(f"| {row['status']} | {row.get('pool_short_name') or row['pool_name']} | {role} | {review_frequency} | {target} | {row.get('source_summary') or '-'} | {missing} |")
    lines.extend(
        [
            "",
            "本摘要為 AI 輔助股票池觀察輸出，不是投資建議；正式用途仍需搭配策略規則、交易成本、資料完整性與風險檢查。",
        ]
    )
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


def _draw_observation_summary_pdf_page(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    generated_count = len(manifest.get("generated", []))
    skipped_count = len(manifest.get("skipped", []))
    consensus = manifest.get("consensus") or {}
    top_label = consensus.get("winner_display") or "模型分歧"
    consensus_state = consensus.get("result_state") or "no_vote"
    consensus_reason = consensus.get("reason") or "尚未產生三池表決結果"

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
    cards = [
        ("已產出股票池", f"{generated_count}", "#13795b"),
        ("跳過股票池", f"{skipped_count}", "#b42318" if skipped_count else "#13795b"),
        ("三池共識", top_label, "#2457a7" if consensus_state == "consensus" else "#b42318"),
        ("使用邊界", "觀察，不是建議", "#17212a"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.225
        ax.add_patch(
            plt.Rectangle((x, 0.74), 0.2, 0.085, facecolor="white", edgecolor="#d9e0e5", linewidth=1, transform=ax.transAxes)
        )
        ax.text(x + 0.014, 0.795, label, color="#66737d", fontsize=9.5, transform=ax.transAxes)
        ax.text(x + 0.014, 0.767, _compact_display(str(value), limit=18), color=color, fontsize=11.2, fontweight="bold", transform=ax.transAxes)

    ax.text(0.06, 0.69, f"表決原因：{consensus_reason[:48]}", color="#52616b", fontsize=10, transform=ax.transAxes)
    _draw_consensus_decision_panel(ax, manifest, rows)
    ax.text(0.06, 0.17, "使用邊界", color="#17212a", fontsize=14, fontweight="bold", transform=ax.transAxes)
    notes = [
        "三池共識只統計正式候選與市場曝險工具；觀察排名不納入票數。",
        "個股若未通過池內攻擊條件，只能列為觀察排名；ETF 依市場曝險工具邏輯顯示。",
        "本報告固定使用同一套資料口徑與股票池設定，方便後續追蹤模型是否穩定。",
        "本報告為 AI 輔助市場觀察與回測工作流輸出，不是投資建議。",
    ]
    for index, note in enumerate(notes):
        ax.text(0.075, 0.126 - index * 0.026, f"• {note}", color="#4d5b66", fontsize=9.6, transform=ax.transAxes)
    ax.text(0.06, 0.026, f"{REPORT_TITLE} · {manifest.get('signal_date', '')}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.026, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _draw_observation_detail_pdf_page(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color="#17212a", transform=ax.transAxes))
    ax.text(0.06, 0.958, "三池前三名與程式判斷原因", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.06,
        0.92,
        f"訊號日 {manifest.get('signal_date', '')} · 表格頁 · {REPORT_VERSION}",
        color="#c8d5df",
        fontsize=10.5,
        transform=ax.transAxes,
    )
    bottom_y = _draw_pool_top3_sections(ax, rows, start_y=0.84)
    reminder_y = max(min(bottom_y - 0.018, 0.075), 0.062)
    ax.text(0.06, reminder_y, "提醒：表格列出的是各池內部排序與程式原因；正式總結以第一頁三池表決為準。", color="#52616b", fontsize=8.6, transform=ax.transAxes)
    ax.text(0.06, 0.04, f"{REPORT_TITLE} · {manifest.get('signal_date', '')}", color="#9aa7b1", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.94, 0.04, "AI_stock_backtest_lab", color="#9aa7b1", fontsize=8.5, ha="right", transform=ax.transAxes)


def _draw_consensus_decision_panel(ax, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    generated_rows = [row for row in rows if row["status"] == "generated"][:3]
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
    ax.text(panel_x + 0.018, panel_y + panel_h - 0.032, "三池表決總覽", color="#17212a", fontsize=12.0, fontweight="bold", transform=ax.transAxes)
    ax.text(panel_x + panel_w - 0.018, panel_y + panel_h - 0.032, "先看結論，再看三池來源", color="#52616b", fontsize=9.2, ha="right", transform=ax.transAxes)
    consensus = manifest.get("consensus") or {}
    winner = consensus.get("winner_ticker")
    state = consensus.get("result_state")
    center_label = consensus.get("winner_display") or "三方分歧"
    reason = consensus.get("reason") or "尚未形成明確共識。"

    winner_color = VOTE_WINNER_COLOR if state == "consensus" else "#2457a7"
    ax.add_patch(
        plt.Rectangle(
            (panel_x + 0.03, panel_y + 0.19),
            panel_w - 0.06,
            0.075,
            facecolor="#17212a",
            edgecolor=winner_color,
            linewidth=1.8,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    ax.add_patch(plt.Rectangle((panel_x + 0.03, panel_y + 0.19), 0.012, 0.075, facecolor=winner_color, edgecolor=winner_color, transform=ax.transAxes, zorder=4))
    ax.text(panel_x + 0.055, panel_y + 0.236, "表決結論", color="#c8d5df", fontsize=8.8, transform=ax.transAxes, zorder=4)
    ax.text(panel_x + 0.055, panel_y + 0.209, _compact_display(center_label, limit=24), color="white", fontsize=13.2, fontweight="bold", transform=ax.transAxes, zorder=4)
    ax.text(panel_x + panel_w - 0.04, panel_y + 0.224, _compact_display(reason, limit=28), color="#d6e1e8", fontsize=9.2, ha="right", transform=ax.transAxes, zorder=4)

    card_specs = [(0.095, 0.435), (0.385, 0.435), (0.675, 0.435)]
    for index, row in enumerate(generated_rows):
        x, y = card_specs[index]
        color = _vote_color(row, consensus, index)
        badge = "共識票" if state == "consensus" and row.get("top_ticker") == winner else ("分歧票" if state != "consensus" else "少數票")
        is_winner = state == "consensus" and row.get("top_ticker") == winner
        fill = "#f4fbf8" if is_winner else "#fff8ef"
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
        eligible = bool(row.get("eligible_for_pool_selection", False))
        top_text = row.get("top_display") or row.get("top_ticker") or ("從缺" if not eligible else "無")
        ax.text(
            x + 0.022,
            y + 0.05,
            _compact_display(top_text, limit=16),
            color=color,
            fontsize=9.3,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=4,
        )
        ax.text(
            x + 0.022,
            y + 0.028,
            _compact_display(_selection_label(str(row.get("selection_layer") or "")) if eligible else "從缺/觀察", limit=14),
            color="#52616b",
            fontsize=8.1,
            transform=ax.transAxes,
            zorder=4,
        )
        ax.text(
            x + 0.022,
            y + 0.01,
            badge,
            color="#7b8994",
            fontsize=7.4,
            transform=ax.transAxes,
            zorder=4,
        )


def _draw_pool_top3_sections(ax, rows: list[dict[str, Any]], *, start_y: float = 0.635) -> float:
    x0 = 0.06
    y = start_y
    generated_rows = [row for row in rows if row["status"] == "generated"]
    skipped_rows = [row for row in rows if row["status"] != "generated"]
    for row in generated_rows[:3]:
        ax.add_patch(plt.Rectangle((x0, y - 0.021), 0.88, 0.032, facecolor="#e9f0f5", edgecolor="#d7e0e7", transform=ax.transAxes))
        title = row.get("pool_short_name") or _short_pool_name(row)
        ax.text(x0 + 0.012, y - 0.011, _compact_display(title, limit=18), color="#17212a", fontsize=10.8, fontweight="bold", transform=ax.transAxes)
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
            ax.text(x0 + 0.012, y + 0.008, _compact_display(role_line, limit=42), color="#52616b", fontsize=7.2, transform=ax.transAxes)
            y -= 0.012
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
            row_h = 0.034
            ax.add_patch(plt.Rectangle((x0, y), 0.88, row_h, facecolor=fill, edgecolor="#e1e7ec", transform=ax.transAxes))
            display = _normalize_display_label(str(candidate.get("display", "")), str(candidate.get("ticker", "")))
            reason = _compact_display(str(candidate.get("reason", "")), limit=30)
            cells = (
                str(candidate.get("rank", "")),
                _compact_display(str(candidate.get("selection_label") or ""), limit=5),
                _compact_display(display, limit=12),
                f"{float(candidate.get('score') or 0):.4f}",
            )
            x = x0
            for cell, width in zip(cells, widths):
                ax.text(x + 0.008, y + row_h - 0.018, cell, color="#26323b", fontsize=7.8, transform=ax.transAxes)
                x += width
            reason_x = x0 + sum(widths[:-1]) + 0.008
            ax.text(reason_x, y + row_h - 0.018, reason, color="#26323b", fontsize=7.4, transform=ax.transAxes)
            y -= row_h
        missing = row.get("missing_price_tickers") or ""
        source = row.get("source_summary") or ""
        if missing or source:
            note = "；".join(part for part in [f"來源：{source}" if source else "", f"缺價：{missing}" if missing else ""] if part)
            ax.text(x0 + 0.008, y - 0.004, _compact_display(note, limit=62), color="#66737d", fontsize=7.1, transform=ax.transAxes)
            y -= 0.018
        y -= 0.012
    for row in skipped_rows[:3]:
        if y < 0.09:
            break
        ax.text(x0, y, f"跳過：{row['pool_name']}，原因：{row['reason']}", color="#b42318", fontsize=8.0, transform=ax.transAxes)
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
    for rank, candidate in enumerate(candidates[:limit], start=1):
        asset_type = _asset_type_for_ticker(candidate.ticker, asset_type_by_ticker)
        selection_layer = _selection_layer_for_candidate(candidate, asset_type)
        rows.append(
            {
                "rank": rank,
                "strength_rank": strength_rank_by_ticker.get(candidate.ticker, rank),
                "ticker": candidate.ticker,
                "display": _candidate_display(observation, candidate.ticker),
                "asset_type": asset_type,
                "score": round(candidate.score, 6),
                "passed": candidate.passed,
                "attack_gate_open": _attack_gate_open_for_candidate(
                    candidate,
                    asset_type,
                    attack_gate_active=(observation.source_metadata or {}).get("attack_gate_active"),
                ),
                "eligible_for_pool_selection": _eligible_for_pool_selection(candidate, asset_type),
                "selection_layer": selection_layer,
                "selection_label": _selection_label(selection_layer),
                "is_model_target": candidate.ticker == observation.top_ticker,
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
        ("大型核心權值股池", "核心防守池"),
        ("核心防守風格池", "核心防守池"),
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
    top_ticker = signal.target_ticker if signal.target_is_actionable else None
    display_by_ticker = {
        symbol["ticker"]: symbol.get("display") or labels.get(symbol["ticker"], symbol["ticker"])
        for symbol in pool.get("resolved_symbols", [])
    }
    top_asset_type = asset_types.get(top_ticker) if top_ticker else None
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
        action_state=signal.model_target_status,
        candidates=candidates,
        top_asset_type=top_asset_type,
        attack_gate_open=_attack_gate_open_for_candidate(
            next((candidate for candidate in candidates if candidate.ticker == top_ticker), None),
            top_asset_type,
            attack_gate_active=signal.attack_gate_active,
        ) if top_ticker else None,
        eligible_for_pool_selection=bool(top_ticker),
        selection_layer=_selection_layer_for_candidate(
            next((candidate for candidate in candidates if candidate.ticker == top_ticker), None),
            top_asset_type,
        ) if top_ticker else SELECTION_NO_SELECTION,
        selection_reason=_selection_reason_for_candidate(
            next((candidate for candidate in candidates if candidate.ticker == top_ticker), None),
            top_asset_type,
        ) if top_ticker else "正式模型目前沒有可投票的池內目標。",
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
            missing.append(ticker)
    return prices, missing


def _price_start_for_pool(pool: dict[str, Any], warmup_start: str) -> str:
    if pool.get("strategy_preset") in {"best_v20260605", "ai_theme_large_cap_v20260613"}:
        return (pd.Timestamp(warmup_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    return warmup_start


def _resolve_dynamic_observation_pool(
    pool: dict[str, Any],
    *,
    signal_date: str,
    radar_snapshot_dir: str | Path | None,
    radar_data_dir: str | Path | None,
    radar_top_n: int,
    tw50_constituents_path: str | Path | None,
) -> dict[str, Any]:
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
