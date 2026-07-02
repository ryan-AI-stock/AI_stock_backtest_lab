from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    POOL1_TICKERS,
    TW50_BENCHMARK,
    _load_price_source_registry,
)
from backtest_lab.data import load_price_csv, split_adjusted_dividends
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_dynamic_adapter_2022_equivalence import _load_required_prices
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant, simulate_regime_mode_switch
from backtest_lab.stock_pool_observation import FROZEN_BEST_GROUP_ID


TASK_ID = "TASK-BACKTEST-CORE-LONG-RANGE-DATA-COMPLETION-CONTINUE-20260702"
DEFAULT_DYNAMIC_UNIVERSE_DIR = "outputs/date_aware_dynamic_universe_state_replay_201411_202112_20260702"
DEFAULT_POOL1_PREVIOUS_DIR = "outputs/pool1_full_state_replay_201411_202112_dynamic_universe_20260702"
DEFAULT_POOL2_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_OUTPUT_DIR = "outputs/long_range_data_completion_continue_20260702"
DEFAULT_START_DATE = "2014-11-03"
DEFAULT_END_DATE = "2021-12-31"


def run_long_range_data_completion_continue(
    *,
    dynamic_universe_dir: str | Path = DEFAULT_DYNAMIC_UNIVERSE_DIR,
    pool1_previous_dir: str | Path = DEFAULT_POOL1_PREVIOUS_DIR,
    pool2_panel_dir: str | Path = DEFAULT_POOL2_PANEL_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    attempt_7_ticker_static_segment: bool = False,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        config = load_config(config_path)
        registry = _load_price_source_registry(price_source_registry)

        log("load_inputs", "started", "")
        dynamic_coverage = pd.read_csv(Path(dynamic_universe_dir) / "dynamic_universe_state_replay_coverage.csv").fillna("")
        dynamic_coverage = dynamic_coverage[
            dynamic_coverage["signal_date"].astype(str).between(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        ].copy()
        pool2_panel = pd.read_csv(Path(pool2_panel_dir) / "pool2_daily_confirmation_panel.csv").fillna("")
        previous_pool1 = _load_optional_csv(Path(pool1_previous_dir) / "pool1_full_state_replayed_signals.csv")

        log("load_prices", "started", "")
        prices, price_meta = _load_required_prices(
            price_cache_dir=price_cache_dir,
            registry=registry,
            required_tickers=sorted(set(POOL1_TICKERS) | {TW50_BENCHMARK}),
        )
        prices, price_meta = _extend_benchmark_with_best_local_source(
            prices=prices,
            price_meta=price_meta,
            price_cache_dir=price_cache_dir,
            benchmark=TW50_BENCHMARK,
        )

        log("build_dynamic_state_contract", "started", "")
        contract = _dynamic_state_contract(dynamic_coverage)
        contract.to_csv(output / "dynamic_universe_state_contract.csv", index=False, encoding="utf-8-sig")

        log("attempt_pool1_dynamic_segments", "started", "")
        pool1_replay, pool1_blocked, segment_attempts = _attempt_pool1_segment_replays(
            dynamic_coverage=dynamic_coverage,
            previous_pool1=previous_pool1,
            prices=prices,
            config=config,
            start=start,
            end=end,
            attempt_7_ticker_static_segment=attempt_7_ticker_static_segment,
        )
        pool1_replay.to_csv(output / "pool1_full_state_replay_201411_202112.csv", index=False, encoding="utf-8-sig")
        pool1_blocked.to_csv(output / "remaining_blocked_rows.csv", index=False, encoding="utf-8-sig")
        segment_attempts.to_csv(output / "pool1_segment_replay_attempts.csv", index=False, encoding="utf-8-sig")

        log("reconstruct_pool2_persistence", "started", "")
        pool2_reconstruction, pool2_blockers = _pool2_persistence_reconstruction(pool2_panel)
        pool2_reconstruction.to_csv(output / "pool2_persistence_reconstruction.csv", index=False, encoding="utf-8-sig")
        pool2_blockers.to_csv(output / "blocker_by_pool2_field.csv", index=False, encoding="utf-8-sig")

        log("build_combined_status", "started", "")
        combined, combined_blockers = _combined_stream_or_blocker(pool1_replay, pool1_blocked, pool2_reconstruction, pool2_blockers)
        combined.to_csv(output / "combined_formal_target_stream_201411_202112.csv", index=False, encoding="utf-8-sig")
        combined_blockers.to_csv(output / "blocker_by_combined_field.csv", index=False, encoding="utf-8-sig")
        source_decision = _source_decision(price_meta, pool1_replay, pool1_blocked, pool2_reconstruction, pool2_blockers)
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")

        log("write_summaries", "started", "")
        handoff = _next_step_handoff(pool1_replay, pool1_blocked, pool2_reconstruction, pool2_blockers, combined_blockers)
        (output / "next_step_handoff.md").write_text(handoff, encoding="utf-8")
        (output / "final_summary_zh.md").write_text(
            _final_summary(pool1_replay, pool1_blocked, pool2_reconstruction, pool2_blockers, combined_blockers),
            encoding="utf-8",
        )

        full_pool1_ready = bool(pool1_blocked.empty and not pool1_replay.empty)
        pool2_ready = bool(pool2_blockers.empty and not pool2_reconstruction.empty)
        combined_ready = bool(combined_blockers.empty and not combined.empty)
        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed" if combined_ready else "completed_partial_precise_blocker",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
            "pool1_rows": int(len(pool1_replay)),
            "pool1_blocked_rows": int(len(pool1_blocked)),
            "pool1_full_period_formal_ready": full_pool1_ready,
            "pool2_rows": int(len(pool2_reconstruction)),
            "pool2_blocker_rows": int(len(pool2_blockers)),
            "pool2_full_period_formal_ready": pool2_ready,
            "combined_rows": int(len(combined)),
            "combined_formal_ready": combined_ready,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "no_target_cash_all_applied_to_2014_2021": False,
            "proxy_used_as_formal": False,
            "candidate_universe_fallback_separation_api_available": True,
            "candidate_universe_fallback_separation_attempted": attempt_7_ticker_static_segment,
            "next_required_task": _next_task(full_pool1_ready, pool2_ready, combined_ready),
            "outputs": {
                "dynamic_contract": "dynamic_universe_state_contract.csv",
                "pool1_replay": "pool1_full_state_replay_201411_202112.csv",
                "remaining_blocked": "remaining_blocked_rows.csv",
                "pool2_reconstruction": "pool2_persistence_reconstruction.csv",
                "pool2_blockers": "blocker_by_pool2_field.csv",
                "combined_stream": "combined_formal_target_stream_201411_202112.csv",
                "combined_blockers": "blocker_by_combined_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_long_range_data_completion_continue", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _dynamic_state_contract(dynamic_coverage: pd.DataFrame) -> pd.DataFrame:
    frame = dynamic_coverage.copy()
    frame["available_universe_count_num"] = pd.to_numeric(frame["available_universe_count"], errors="coerce").fillna(0).astype(int)
    grouped = (
        frame.groupby("available_universe_count_num")
        .agg(first_signal_date=("signal_date", "min"), last_signal_date=("signal_date", "max"), rows=("signal_date", "count"))
        .reset_index()
    )
    rows = []
    for item in grouped.to_dict(orient="records"):
        count = int(item["available_universe_count_num"])
        rows.append(
            {
                "available_universe_count": count,
                "first_signal_date": item["first_signal_date"],
                "last_signal_date": item["last_signal_date"],
                "rows": int(item["rows"]),
                "state_contract": _state_contract_for_count(count),
                "formal_ready_candidate": count >= 7,
            }
        )
    return pd.DataFrame(rows)


def _attempt_pool1_segment_replays(
    *,
    dynamic_coverage: pd.DataFrame,
    previous_pool1: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    attempt_7_ticker_static_segment: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    blocked_parts: list[pd.DataFrame] = []
    attempts: list[dict[str, Any]] = []
    coverage = dynamic_coverage.copy()
    coverage["count"] = pd.to_numeric(coverage["available_universe_count"], errors="coerce").fillna(0).astype(int)

    seven = coverage[coverage["count"].eq(7)]
    if not seven.empty:
        seven_start = pd.Timestamp(str(seven["signal_date"].min())).normalize()
        seven_end = pd.Timestamp(str(seven["signal_date"].max())).normalize()
        seven_tickers = _candidate_tickers_from_row(seven.iloc[0])
        if attempt_7_ticker_static_segment:
            try:
                seven_replay = _run_pool1_static_subset_replay(
                    prices=prices,
                    config=config,
                    candidate_tickers=seven_tickers,
                    start_date=seven_start,
                    end_date=seven_end,
                    segment_id="dynamic_7_ticker_segment",
                    separate_candidate_universe=True,
                )
                rows.append(seven_replay)
                attempts.append(
                    {
                        "segment_id": "dynamic_7_ticker_segment",
                        "start_date": seven_start.strftime("%Y-%m-%d"),
                        "end_date": seven_end.strftime("%Y-%m-%d"),
                        "candidate_tickers": "|".join(seven_tickers),
                        "status": "completed_segment_replay",
                        "rows": len(seven_replay),
                        "formal_ready_for_segment": True,
                        "blocker": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised by real data if a source breaks
                blocked_parts.append(_block_rows(seven, "pool1_7_ticker_segment_replay_failed", str(exc)))
                attempts.append(
                    {
                        "segment_id": "dynamic_7_ticker_segment",
                        "start_date": seven_start.strftime("%Y-%m-%d"),
                        "end_date": seven_end.strftime("%Y-%m-%d"),
                        "candidate_tickers": "|".join(seven_tickers),
                        "status": "failed_segment_replay",
                        "rows": 0,
                        "formal_ready_for_segment": False,
                        "blocker": str(exc),
                    }
                )
        else:
            reason = (
                "Requires a dynamic Pool1 state-machine adapter that separates candidate universe from "
                "0050 fallback/benchmark inputs and exposes state injection/carryover. The current "
                "simulate_regime_mode_switch static subset route is intentionally not used as formal."
            )
            blocked_parts.append(_block_rows(seven, "pool1_7_ticker_dynamic_state_adapter_missing", reason))
            attempts.append(
                {
                    "segment_id": "dynamic_7_ticker_segment",
                    "start_date": seven_start.strftime("%Y-%m-%d"),
                    "end_date": seven_end.strftime("%Y-%m-%d"),
                    "candidate_tickers": "|".join(seven_tickers),
                    "status": "blocked_requires_dynamic_state_adapter",
                    "rows": 0,
                    "formal_ready_for_segment": False,
                    "blocker": reason,
                }
            )

    if not previous_pool1.empty:
        prior = previous_pool1.copy()
        prior["segment_id"] = "dynamic_8_ticker_segment"
        prior["segment_source"] = DEFAULT_POOL1_PREVIOUS_DIR
        rows.append(prior)
        attempts.append(
            {
                "segment_id": "dynamic_8_ticker_segment",
                "start_date": str(prior["signal_date"].min()),
                "end_date": str(prior["signal_date"].max()),
                "candidate_tickers": "|".join(POOL1_TICKERS),
                "status": "accepted_previous_formal_ready_segment",
                "rows": len(prior),
                "formal_ready_for_segment": True,
                "blocker": "",
            }
        )

    early = coverage[coverage["count"].lt(7)]
    if not early.empty:
        blocked_parts.append(
            _block_rows(
                early,
                "pool1_dynamic_universe_warmup_or_lifecycle_not_ready",
                "available universe has fewer than 7 Pool1 tickers; current Pool1 state machine cannot produce formal attack-gate state without enough date-aware candidates and warmup history.",
            )
        )

    replay = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not replay.empty:
        replay = replay.sort_values("signal_date").drop_duplicates("signal_date", keep="last").reset_index(drop=True)
    blocked = pd.concat(blocked_parts, ignore_index=True, sort=False) if blocked_parts else pd.DataFrame()
    if not blocked.empty:
        replay_dates = set(replay["signal_date"].astype(str)) if not replay.empty else set()
        blocked = blocked[~blocked["signal_date"].astype(str).isin(replay_dates)].sort_values("signal_date").reset_index(drop=True)

    if len(attempts) >= 2:
        attempts.append(
            {
                "segment_id": "cross_segment_state_carryover",
                "start_date": "",
                "end_date": "",
                "candidate_tickers": "",
                "status": "blocked_for_full_period_formal_ready",
                "rows": 0,
                "formal_ready_for_segment": False,
                "blocker": "simulate_regime_mode_switch does not expose state injection/carryover API; 7-ticker and 8-ticker segments are valid segment replays but not one continuous formal state stream.",
            }
        )
    return replay, blocked, pd.DataFrame(attempts)


def _extend_benchmark_with_best_local_source(
    *,
    prices: dict[str, pd.DataFrame],
    price_meta: dict[str, dict[str, Any]],
    price_cache_dir: str | Path,
    benchmark: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    current = prices.get(benchmark)
    if current is None:
        return prices, price_meta
    cache_root = Path(price_cache_dir)
    file_name = f"{benchmark.replace('.', '_')}.csv"
    candidates: list[tuple[pd.Timestamp, int, Path, pd.DataFrame]] = []
    search_roots = [cache_root]
    if cache_root.name != "backtest_cache":
        search_roots.append(cache_root.parent)
    seen: set[Path] = set()
    for root in search_roots:
        for path in root.rglob(file_name):
            if path in seen:
                continue
            seen.add(path)
            try:
                frame = load_price_csv(path)
            except Exception:
                continue
            if frame.empty:
                continue
            candidates.append((pd.Timestamp(frame.index.min()), len(frame), path, frame))
    if not candidates:
        return prices, price_meta
    best_start, _rows, best_path, best_frame = sorted(candidates, key=lambda item: (item[0], -item[1]))[0]
    current_start = pd.Timestamp(current.index.min())
    if best_start >= current_start:
        return prices, price_meta
    updated_prices = dict(prices)
    updated_meta = dict(price_meta)
    updated_prices[benchmark] = best_frame
    meta = dict(updated_meta.get(benchmark, {}))
    meta.update(
        {
            "source": str(best_path),
            "source_type": "local_best_benchmark_overlay",
            "overlay_reason": "earlier 0050 history is required for market regime warmup in 2015 dynamic Pool1 segment replay",
            "first_date": best_start.strftime("%Y-%m-%d"),
            "last_date": pd.Timestamp(best_frame.index.max()).strftime("%Y-%m-%d"),
            "rows": int(len(best_frame)),
        }
    )
    updated_meta[benchmark] = meta
    return updated_prices, updated_meta


def _run_pool1_static_subset_replay(
    *,
    prices: dict[str, pd.DataFrame],
    config: Any,
    candidate_tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    segment_id: str,
    separate_candidate_universe: bool,
) -> pd.DataFrame:
    group = config.group_by_id(FROZEN_BEST_GROUP_ID)
    group_assets = {asset.ticker: asset for asset in group.assets}
    required = sorted(set(candidate_tickers) | {TW50_BENCHMARK})
    labels = {ticker: group_assets[ticker].label for ticker in required if ticker in group_assets}
    asset_types = {ticker: group_assets[ticker].asset_type for ticker in required if ticker in group_assets}
    variant = frozen_cycle_proven_top1_v1_variant()
    dividends = {
        ticker: split_adjusted_dividends(prices[ticker], config.manual_splits.get(ticker, ()))
        for ticker in required
        if ticker in prices
    }
    result = simulate_regime_mode_switch(
        name=f"pool1_{segment_id}",
        prices_by_ticker={ticker: prices[ticker] for ticker in required if ticker in prices},
        asset_types=asset_types,
        market_prices=prices[TW50_BENCHMARK],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_cash=config.initial_cash_twd,
        cost_model=config.cost_model,
        variant=variant,
        dividend_series_by_ticker=dividends,
        candidate_universe_tickers=tuple(candidate_tickers) if separate_candidate_universe else None,
    )
    equity = result.equity_curve.reset_index().rename(columns={"date": "signal_date"})
    rows: list[dict[str, Any]] = []
    for item in equity.to_dict(orient="records"):
        signal_date = pd.Timestamp(item["signal_date"]).strftime("%Y-%m-%d")
        target = str(item.get("current_ticker") or "")
        exposure = _to_float(item.get("current_exposure")) or 0.0
        actionable = bool(target and target.lower() != "cash" and exposure > 0)
        rows.append(
            {
                "signal_date": signal_date,
                "pool1_target": target if actionable else "",
                "pool1_target_display": labels.get(target, target) if actionable else "",
                "pool1_target_weights": json.dumps({target: 1.0} if target and actionable else {}, ensure_ascii=False),
                "attack_gate_active": _bool_like(item.get("attack_gate_active")),
                "attack_gate_ever_activated": _bool_like(item.get("attack_gate_ever_activated")),
                "risk_off_active": _bool_like(item.get("risk_off_active")),
                "target_is_actionable": actionable,
                "model_target_status": "has_formal_pool1_target" if actionable else "no_actionable_pool1_target",
                "mode": str(item.get("mode") or ""),
                "regime": str(item.get("regime") or ""),
                "current_exposure": round(exposure, 8),
                "available_universe_count": len(candidate_tickers),
                "candidate_tickers": "|".join(candidate_tickers),
                "segment_id": segment_id,
                "segment_source": "simulate_regime_mode_switch_static_subset",
                "candidate_universe_fallback_separated": separate_candidate_universe,
                "source_formal_ready": True,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _block_rows(frame: pd.DataFrame, blocker: str, reason: str) -> pd.DataFrame:
    rows = []
    for item in frame.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "available_universe_count": item.get("available_universe_count", ""),
                "candidate_tickers": str(item.get("candidate_tickers") or ""),
                "blocker": blocker,
                "reason": reason,
                "source_formal_ready": False,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _pool2_persistence_reconstruction(pool2_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pool2_panel.copy()
    frame["eligible"] = frame["eligible_for_pool_selection"].map(_bool_like)
    frame["date"] = frame["date"].astype(str)
    eligible = frame[frame["eligible"]].sort_values(["date", "rank", "raw_rank"], na_position="last")
    if eligible.empty:
        blockers = pd.DataFrame(
            [
                {
                    "field_name": "pool2_persistence_full_reconstruction",
                    "blocker": "no_eligible_pool2_rows_201411_202112",
                    "affected_period": f"{frame['date'].min()}..{frame['date'].max()}" if not frame.empty else "",
                    "detail": "Existing Pool2 daily confirmation panel has zero eligible_for_pool_selection=true rows, so Pool2 cannot confirm or persist any candidate in 2014-2021.",
                    "next_action": "Reconstruct Pool2 persistence/confirmation contract from source features; current panel is diagnostic-only for this interval.",
                    "formal_ready": False,
                }
            ]
        )
        return pd.DataFrame(), blockers

    best = eligible.groupby("date", as_index=False).first()
    rows = []
    for item in best.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": str(item.get("date") or ""),
                "pool2_target": str(item.get("candidate_ticker") or ""),
                "pool2_target_display": f"{item.get('candidate_name','')}({item.get('candidate_ticker','')})",
                "pool2_confirmation_state": str(item.get("confirmation_state") or ""),
                "pool2_market_exposure_support": str(item.get("market_exposure_support") or ""),
                "score": item.get("score", ""),
                "rank": item.get("rank", ""),
                "source_formal_ready": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(columns=["field_name", "blocker", "affected_period", "detail", "next_action", "formal_ready"])


def _combined_stream_or_blocker(
    pool1_replay: pd.DataFrame,
    pool1_blocked: pd.DataFrame,
    pool2_reconstruction: pd.DataFrame,
    pool2_blockers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blockers = []
    if not pool1_blocked.empty:
        blockers.append(
            {
                "field_name": "pool1_full_period_state_stream",
                "blocker": "pool1_remaining_blocked_rows",
                "affected_period": f"{pool1_blocked['signal_date'].min()}..{pool1_blocked['signal_date'].max()}",
                "detail": "Pool1 is not full-period formal-ready; combined target stream cannot be declared formal-ready.",
                "next_action": "Add state injection/carryover API or accepted continuous dynamic state-machine adapter.",
                "formal_ready": False,
            }
        )
    if not pool2_blockers.empty:
        blockers.extend(pool2_blockers.to_dict(orient="records"))
    if blockers:
        return _empty_combined_stream(), pd.DataFrame(blockers)

    merged = pool1_replay.merge(pool2_reconstruction, on="signal_date", how="left")
    rows = []
    for item in merged.fillna("").to_dict(orient="records"):
        target = str(item.get("pool1_target") or "")
        weights = str(item.get("pool1_target_weights") or "{}")
        rows.append(
            {
                "signal_date": str(item.get("signal_date") or ""),
                "formal_target": target,
                "target_weights": weights,
                "pool1_target": target,
                "pool2_target": str(item.get("pool2_target") or ""),
                "pool2_confirmation_state": str(item.get("pool2_confirmation_state") or ""),
                "execution_action_basis": "next_day",
                "risk_off_state": "",
                "source_formal_ready": True,
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(columns=["field_name", "blocker", "affected_period", "detail", "next_action", "formal_ready"])


def _empty_combined_stream() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_date",
            "formal_target",
            "target_weights",
            "pool1_target",
            "pool2_target",
            "pool2_confirmation_state",
            "execution_action_basis",
            "risk_off_state",
            "source_formal_ready",
            "no_target_cash_all_applied",
        ]
    )


def _source_decision(
    price_meta: dict[str, dict[str, Any]],
    pool1_replay: pd.DataFrame,
    pool1_blocked: pd.DataFrame,
    pool2_reconstruction: pd.DataFrame,
    pool2_blockers: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "pool1_segment_replay",
                "status": "partial" if not pool1_blocked.empty else "accepted",
                "formal_or_proxy": "segment_formal_ready_not_full_period" if not pool1_blocked.empty else "formal_ready",
                "decision": "Pool1 segment rows are generated by current simulate_regime_mode_switch without changing thresholds; full-period continuity is blocked if any blocked rows remain.",
                "rows": int(len(pool1_replay)),
            },
            {
                "source_layer": "pool2_persistence",
                "status": "blocked" if not pool2_blockers.empty else "accepted",
                "formal_or_proxy": "blocked" if not pool2_blockers.empty else "formal_ready",
                "decision": "Pool2 panel must produce eligible persisted confirmation rows before combined formal stream can be built.",
                "rows": int(len(pool2_reconstruction)),
            },
            {
                "source_layer": "price_sources",
                "status": "accepted",
                "formal_or_proxy": "formal_input_for_segment_replay",
                "decision": "Price inputs were used only through existing loaders; no proxy was promoted to formal.",
                "rows": len(price_meta),
                "metadata": json.dumps(price_meta, ensure_ascii=False),
            },
        ]
    )


def _next_step_handoff(
    pool1_replay: pd.DataFrame,
    pool1_blocked: pd.DataFrame,
    pool2_reconstruction: pd.DataFrame,
    pool2_blockers: pd.DataFrame,
    combined_blockers: pd.DataFrame,
) -> str:
    lines = [
        "# Long-range data completion handoff",
        "",
        "## Completed in this pass",
        f"- Pool1 replay rows: {len(pool1_replay)}",
        f"- Pool1 remaining blocked rows: {len(pool1_blocked)}",
        f"- Pool2 reconstructed rows: {len(pool2_reconstruction)}",
        f"- Pool2 blocker rows: {len(pool2_blockers)}",
        "",
        "## Current decision",
    ]
    if combined_blockers.empty:
        lines.append("- Combined formal target stream is ready for Experiments next-day replay validation.")
    else:
        lines.append("- Combined formal target stream is not ready. Do not run long-range formal performance yet.")
        lines.append("")
        lines.append("## Remaining blockers")
        for item in combined_blockers.to_dict(orient="records"):
            lines.append(f"- {item.get('field_name')}: {item.get('blocker')} ({item.get('affected_period')})")
    lines.extend(
        [
            "",
            "## Boundaries",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all was not applied to unvalidated 2014-2021 rows",
        ]
    )
    return "\n".join(lines) + "\n"


def _final_summary(
    pool1_replay: pd.DataFrame,
    pool1_blocked: pd.DataFrame,
    pool2_reconstruction: pd.DataFrame,
    pool2_blockers: pd.DataFrame,
    combined_blockers: pd.DataFrame,
) -> str:
    status = "not_ready_for_long_range_next_day_replay" if not combined_blockers.empty else "ready_for_experiments_replay"
    return "\n".join(
        [
            "# 2014/11-2021 long-range data completion continuation",
            "",
            f"- status: {status}",
            f"- Pool1 replay rows: {len(pool1_replay)}",
            f"- Pool1 blocked rows: {len(pool1_blocked)}",
            f"- Pool2 reconstructed rows: {len(pool2_reconstruction)}",
            f"- Pool2 blocker rows: {len(pool2_blockers)}",
            f"- Combined blocker rows: {len(combined_blockers)}",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied_to_2014_2021=false",
            "",
            "The package advances Pool1 segment replay where the current state machine can be reused, but it does not promote partial rows to a formal long-range target stream.",
        ]
    ) + "\n"


def _state_contract_for_count(count: int) -> str:
    if count < 7:
        return "blocked: insufficient date-aware Pool1 universe for current formal state replay"
    if count == 7:
        return "attemptable: static subset segment excluding not-yet-scoring ticker; cross-segment state carryover still must be validated"
    if count >= len(POOL1_TICKERS):
        return "accepted: all Pool1 tickers scoring-ready; current static state machine can be reused"
    return "blocked: unsupported dynamic universe count"


def _candidate_tickers_from_row(row: pd.Series) -> list[str]:
    raw = str(row.get("candidate_tickers") or "")
    sep = "|" if "|" in raw else ","
    return [ticker.strip() for ticker in raw.split(sep) if ticker.strip()]


def _next_task(pool1_ready: bool, pool2_ready: bool, combined_ready: bool) -> str:
    if combined_ready:
        return "handoff_to_experiments_long_range_next_day_replay_validation"
    if not pool1_ready:
        return "dynamic_universe_state_injection_or_continuous_state_machine_adapter"
    if not pool2_ready:
        return "pool2_persistence_full_reconstruction"
    return "combined_formal_target_stream_builder"


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continue 2014-2021 long-range formal data completion.")
    parser.add_argument("--dynamic-universe-dir", default=DEFAULT_DYNAMIC_UNIVERSE_DIR)
    parser.add_argument("--pool1-previous-dir", default=DEFAULT_POOL1_PREVIOUS_DIR)
    parser.add_argument("--pool2-panel-dir", default=DEFAULT_POOL2_PANEL_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--attempt-7-ticker-static-segment", action="store_true")
    args = parser.parse_args(argv)
    output = run_long_range_data_completion_continue(
        dynamic_universe_dir=args.dynamic_universe_dir,
        pool1_previous_dir=args.pool1_previous_dir,
        pool2_panel_dir=args.pool2_panel_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        config_path=args.config_path,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        attempt_7_ticker_static_segment=args.attempt_7_ticker_static_segment,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
