from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.config import load_config
from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_END_DATE,
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    DEFAULT_START_DATE,
    FORMAL_CANDIDATE_EXCLUDED_TICKERS,
    TW50_BENCHMARK,
    TW50_PERSISTENCE_LOOKBACK,
    TW50_PERSISTENCE_MIN_DAYS,
    TW50_RET20_MIN,
    TW50_RET60_MARGIN,
    TW50_RET60_MIN,
    _anchor_tickers,
    _display_name,
    _empty_tw50_gate,
    _load_price_source,
    _load_price_source_registry,
    _name_map,
    _pool2_feature_frames,
    _pool2_confirmation_state,
    _resolve_anchor_from_frame,
    _ret60_series_by_ticker,
    _score_pool2_candidates_direct,
    _trading_dates,
    _tw50_gate_result,
    _window_return_on_or_before,
)
from backtest_lab.pcf_pit_candidate_adapter import DEFAULT_MONTHLY_ANCHOR_PATH, load_0050_pcf_monthly_anchor


TASK_ID = "TASK-BACKTEST-CORE-POOL2-DATE-BATCHED-PERSISTENCE-RECONSTRUCTION-20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool2_date_batched_persistence_reconstruction_201411_202112_20260702"


def run_pool2_date_batched_persistence_reconstruction(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    batch_size: int = 30,
    max_batches: int | None = None,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    monthly_anchor_path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    pool1_status_output: str | Path = "outputs/long_range_data_completion_continue_checkpointed_20260702",
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        log("load_inputs", "started", f"{start.date()}..{end.date()}")
        config = load_config(config_path)
        names = _name_map(config)
        anchor = load_0050_pcf_monthly_anchor(monthly_anchor_path)
        registry = _load_price_source_registry(price_source_registry)
        trading_dates = _trading_dates(price_cache_dir, start, end)
        batches = _date_batches(trading_dates, batch_size)
        if max_batches is not None:
            batches = batches[:max_batches]

        log("load_price_sources", "started", f"anchor_tickers={anchor['ticker'].nunique()}")
        all_tickers = sorted(_anchor_tickers(anchor) | {TW50_BENCHMARK})
        prices_by_ticker: dict[str, pd.DataFrame] = {}
        price_source_meta: dict[str, dict[str, Any]] = {}
        missing_price_tickers: list[str] = []
        for ticker in all_tickers:
            frame, meta = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
            if frame is None:
                missing_price_tickers.append(ticker)
                continue
            prices_by_ticker[ticker] = frame
            price_source_meta[ticker] = meta

        log("precompute_features", "started", f"priced_tickers={len(prices_by_ticker)}")
        features_by_ticker = _pool2_feature_frames(prices_by_ticker)
        ret60_by_ticker = _ret60_series_by_ticker(prices_by_ticker)
        ret60_frame = _ret60_frame(ret60_by_ticker)
        persistence_cache: dict[str, dict[str, tuple[int, int]]] = {}

        panel_parts: list[pd.DataFrame] = []
        daily_parts: list[pd.DataFrame] = []
        batch_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        for batch_index, batch_dates in enumerate(batches, start=1):
            batch_start = batch_dates[0]
            batch_end = batch_dates[-1]
            log("run_batch", "started", f"{batch_index}/{len(batches)} {batch_start.date()}..{batch_end.date()}")
            try:
                panel, daily = _build_pool2_panel_for_dates(
                    trading_dates=batch_dates,
                    anchor=anchor,
                    prices_by_ticker=prices_by_ticker,
                    names=names,
                    price_source_meta=price_source_meta,
                    features_by_ticker=features_by_ticker,
                    ret60_frame=ret60_frame,
                    persistence_cache=persistence_cache,
                )
                panel_parts.append(panel)
                daily_parts.append(daily)
                batch_rows.append(_batch_summary_row(batch_index, batch_start, batch_end, panel, daily, "completed", ""))
                _write_progress(output, panel_parts, daily_parts, batch_rows, failed_rows)
                log("run_batch", "completed", f"{batch_index}/{len(batches)}")
            except Exception as exc:  # noqa: BLE001 - keep batch failures observable.
                failed_rows.append(
                    {
                        "batch_index": batch_index,
                        "batch_start": batch_start.strftime("%Y-%m-%d"),
                        "batch_end": batch_end.strftime("%Y-%m-%d"),
                        "error": str(exc),
                    }
                )
                batch_rows.append(_batch_summary_row(batch_index, batch_start, batch_end, pd.DataFrame(), pd.DataFrame(), "failed", str(exc)))
                _write_progress(output, panel_parts, daily_parts, batch_rows, failed_rows)
                log("run_batch", "failed", f"{batch_index}/{len(batches)} {exc}")

        panel_all = pd.concat(panel_parts, ignore_index=True, sort=False) if panel_parts else _empty_panel()
        daily_all = pd.concat(daily_parts, ignore_index=True, sort=False) if daily_parts else _empty_daily()
        eligible = panel_all[panel_all.get("eligible_for_pool_selection", pd.Series(dtype=str)).astype(str).str.lower().eq("true")].copy()
        gate_breakdown = _gate_breakdown(panel_all)
        combined_blockers = _combined_readiness_blockers(pool1_status_output, panel_all, eligible, failed_rows, batches, max_batches)

        log("write_final_outputs", "started", str(output))
        panel_all.to_csv(output / "pool2_batched_confirmation_panel.csv", index=False, encoding="utf-8-sig")
        daily_all.to_csv(output / "pool2_daily_vote_status.csv", index=False, encoding="utf-8-sig")
        eligible.to_csv(output / "pool2_reconstructed_eligible_rows.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(batch_rows).to_csv(output / "pool2_batch_summary.csv", index=False, encoding="utf-8-sig")
        gate_breakdown.to_csv(output / "pool2_gate_breakdown.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"ticker": missing_price_tickers}).to_csv(output / "missing_price_tickers.csv", index=False, encoding="utf-8-sig")
        combined_blockers.to_csv(output / "combined_formal_target_stream_readiness.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_handoff_text(panel_all, eligible, combined_blockers, batches, max_batches), encoding="utf-8")

        status = "completed_partial_precise_blocker" if (failed_rows or not combined_blockers.empty or max_batches is not None) else "completed_pool2_reconstruction"
        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": status,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
            "batch_size": batch_size,
            "requested_batches": len(_date_batches(trading_dates, batch_size)),
            "executed_batches": len(batches),
            "max_batches": max_batches,
            "pool2_panel_rows": int(len(panel_all)),
            "pool2_daily_rows": int(len(daily_all)),
            "persistence_passed_rows": int(panel_all.get("persistence_passed", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum()),
            "eligible_for_pool_selection_rows": int(len(eligible)),
            "eligible_signal_dates": int(eligible["date"].nunique()) if not eligible.empty else 0,
            "failed_batches": int(len(failed_rows)),
            "missing_price_ticker_count": int(len(missing_price_tickers)),
            "combined_formal_target_stream_ready": bool(combined_blockers.empty and max_batches is None and not failed_rows),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "outputs": {
                "pool2_panel": "pool2_batched_confirmation_panel.csv",
                "daily_vote_status": "pool2_daily_vote_status.csv",
                "eligible_rows": "pool2_reconstructed_eligible_rows.csv",
                "batch_summary": "pool2_batch_summary.csv",
                "gate_breakdown": "pool2_gate_breakdown.csv",
                "combined_readiness": "combined_formal_target_stream_readiness.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_text(manifest, combined_blockers), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": status, "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(failed_rows, columns=["batch_index", "batch_start", "batch_end", "error"]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _build_pool2_panel_for_dates(
    *,
    trading_dates: list[pd.Timestamp],
    anchor: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    names: dict[str, str],
    price_source_meta: dict[str, dict[str, Any]],
    features_by_ticker: dict[str, pd.DataFrame],
    ret60_frame: pd.DataFrame,
    persistence_cache: dict[str, dict[str, tuple[int, int]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for signal_date in trading_dates:
        constituents, anchor_meta = _resolve_anchor_from_frame(anchor, signal_date)
        anchor_names = {
            f"{str(row.ticker).zfill(4)}.TW": str(row.name).strip()
            for row in constituents[["ticker", "name"]].itertuples(index=False)
        }
        candidate_tickers = list(anchor_names)
        prices = {ticker: prices_by_ticker[ticker] for ticker in candidate_tickers if ticker in prices_by_ticker}
        if TW50_BENCHMARK in prices_by_ticker:
            prices[TW50_BENCHMARK] = prices_by_ticker[TW50_BENCHMARK]
        if not prices:
            daily_rows.append(_daily_row(signal_date, "", False, "no_price_frames_for_pit_constituents", anchor_meta))
            continue
        candidates = _score_pool2_candidates_direct(prices.keys(), signal_date, features_by_ticker)
        gate_by_ticker = _tw50_gate_details_cached(candidates, prices, signal_date, ret60_frame, persistence_cache)
        ranked = sorted(
            candidates,
            key=lambda item: (
                bool(gate_by_ticker.get(item.ticker, {}).get("candidate_support_without_persistence", False)),
                item.score,
                item.ticker,
            ),
            reverse=True,
        )
        eligible_rank = 0
        vote = ""
        eligible_vote = ""
        for raw_rank, candidate in enumerate(ranked, start=1):
            excluded = candidate.ticker in FORMAL_CANDIDATE_EXCLUDED_TICKERS
            gate = gate_by_ticker.get(candidate.ticker, _empty_tw50_gate(candidate))
            support_candidate = bool(gate.get("candidate_support_without_persistence", False)) and not excluded
            eligible = bool(gate.get("eligible_for_pool_selection", False)) and not excluded
            if support_candidate:
                eligible_rank += 1
                if not vote:
                    vote = candidate.ticker
            if eligible and not eligible_vote:
                eligible_vote = candidate.ticker
            meta = price_source_meta.get(candidate.ticker, {})
            rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "pool_id": "tw50_dynamic_constituents_v0_pcf_monthly_anchor_candidate",
                    "pool_name": "大型廣度池",
                    "candidate_ticker": candidate.ticker,
                    "candidate_name": _display_name(candidate.ticker, names, fallback=anchor_names.get(candidate.ticker, "")),
                    "score": round(float(candidate.score), 8),
                    "raw_rank": raw_rank,
                    "rank": eligible_rank if support_candidate else "",
                    "passed": str(bool(candidate.passed)).lower(),
                    "confirmation_state": _pool2_confirmation_state(vote, support_candidate),
                    "market_exposure_support": "supports_market_exposure" if vote else "not_sufficiently_confirmed",
                    "eligible_for_pool_selection": str(eligible).lower(),
                    "candidate_support_without_persistence": str(support_candidate).lower(),
                    "attack_gate_open": str(bool(gate.get("attack_gate_open", False))).lower(),
                    "reason": gate.get("gate_reason", candidate.reason),
                    "base_pool_passed": str(bool(gate.get("base_pool_passed", candidate.passed))).lower(),
                    "benchmark_margin_passed": str(bool(gate.get("benchmark_margin_passed", False))).lower(),
                    "momentum_quality_passed": str(bool(gate.get("momentum_quality_passed", False))).lower(),
                    "persistence_passed": str(bool(gate.get("persistence_passed", False))).lower(),
                    "persistence_days": gate.get("persistence_days", 0),
                    "persistence_total": gate.get("persistence_total", 0),
                    "anchor_effective_month": anchor_meta["effective_month"],
                    "anchor_effective_date": anchor_meta["anchor_effective_date"],
                    "anchor_after_query_date": str(anchor_meta["anchor_after_query_date"]).lower(),
                    "pit_safe_for_query_date": str(anchor_meta["pit_safe_for_query_date"]).lower(),
                    "formal_exact": "false",
                    "source_type": "source_backed_manual_candidate",
                    "price_only_used": str(not bool(meta.get("adjusted_close_available", True))).lower(),
                    "adjusted_close_available": str(bool(meta.get("adjusted_close_available", True))).lower(),
                }
            )
        daily_rows.append(
            _daily_row(
                signal_date,
                eligible_vote,
                bool(eligible_vote),
                "" if eligible_vote else "no_pool2_persistent_eligible_candidate",
                anchor_meta,
                support_vote=vote,
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def _tw50_gate_details_cached(
    candidates: list[Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    ret60_frame: pd.DataFrame,
    persistence_cache: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    benchmark_ret60 = _window_return_on_or_before(prices_by_ticker.get(TW50_BENCHMARK), signal_date, 60)
    date_key = signal_date.strftime("%Y-%m-%d")
    if date_key not in persistence_cache:
        persistence_cache[date_key] = _persistence_counts_for_date(ret60_frame, signal_date)
    for candidate in candidates:
        if benchmark_ret60 is None:
            result[candidate.ticker] = _tw50_gate_result(
                candidate,
                eligible=False,
                benchmark_margin_passed=False,
                momentum_quality_passed=False,
                persistence_passed=False,
                ret60_margin=None,
                persistence_days=0,
                persistence_total=0,
                reason_prefix="缺少 0050 benchmark 價格，不能確認相對超額",
            )
            continue
        ret60_margin = candidate.ret60 - benchmark_ret60
        benchmark_margin_passed = ret60_margin >= TW50_RET60_MARGIN
        momentum_quality_passed = candidate.ret20 >= TW50_RET20_MIN and candidate.ret60 >= TW50_RET60_MIN and candidate.ret60 > 0
        persistence_days, persistence_total = persistence_cache[date_key].get(candidate.ticker, (0, 0))
        persistence_passed = persistence_total >= TW50_PERSISTENCE_LOOKBACK and persistence_days >= TW50_PERSISTENCE_MIN_DAYS
        support_without_persistence = bool(candidate.passed and benchmark_margin_passed and momentum_quality_passed)
        eligible = bool(support_without_persistence and persistence_passed)
        gate = _tw50_gate_result(
            candidate,
            eligible=eligible,
            support_without_persistence=support_without_persistence,
            benchmark_margin_passed=benchmark_margin_passed,
            momentum_quality_passed=momentum_quality_passed,
            persistence_passed=persistence_passed,
            ret60_margin=ret60_margin,
            persistence_days=persistence_days,
            persistence_total=persistence_total,
            reason_prefix="",
        )
        gate["persistence_days"] = persistence_days
        gate["persistence_total"] = persistence_total
        result[candidate.ticker] = gate
    return result


def _ret60_frame(ret60_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    if not ret60_by_ticker:
        return pd.DataFrame()
    return pd.concat(ret60_by_ticker, axis=1, sort=False).sort_index()


def _persistence_counts_for_date(ret60_frame: pd.DataFrame, signal_date: pd.Timestamp) -> dict[str, tuple[int, int]]:
    if ret60_frame.empty or TW50_BENCHMARK not in ret60_frame.columns:
        return {}
    passed: dict[str, int] = {}
    evaluated = 0
    tickers = [str(ticker) for ticker in ret60_frame.columns if ticker != TW50_BENCHMARK]
    benchmark_dates = list(
        ret60_frame.index[(ret60_frame.index <= signal_date) & ret60_frame[TW50_BENCHMARK].notna()]
    )[-TW50_PERSISTENCE_LOOKBACK:]
    history = ret60_frame.ffill().loc[benchmark_dates]
    for _, row in history.iterrows():
        benchmark_ret60 = row.get(TW50_BENCHMARK)
        if pd.isna(benchmark_ret60):
            continue
        candidates = row.drop(labels=[TW50_BENCHMARK], errors="ignore").dropna()
        if candidates.empty:
            continue
        evaluated += 1
        top_cutoff = max(5, int(len(candidates) * 0.2 + 0.9999))
        top = candidates.sort_values(ascending=False).head(top_cutoff)
        for ticker, ret60 in top.items():
            if ret60 - benchmark_ret60 >= TW50_RET60_MARGIN:
                passed[ticker] = passed.get(ticker, 0) + 1
    return {ticker: (passed.get(ticker, 0), evaluated) for ticker in tickers}


def _date_batches(trading_dates: list[pd.Timestamp], batch_size: int) -> list[list[pd.Timestamp]]:
    return [trading_dates[offset : offset + batch_size] for offset in range(0, len(trading_dates), batch_size)]


def _daily_row(
    signal_date: pd.Timestamp,
    vote: str,
    ready: bool,
    blocker: str,
    anchor_meta: dict[str, Any],
    *,
    support_vote: str = "",
) -> dict[str, Any]:
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "pool2_vote": vote,
        "pool2_support_without_persistence_vote": support_vote,
        "pool2_confirmation_ready": str(ready).lower(),
        "pool2_blocker": blocker,
        "anchor_after_query_date": str(anchor_meta.get("anchor_after_query_date", False)).lower(),
        "pit_safe_for_query_date": str(anchor_meta.get("pit_safe_for_query_date", False)).lower(),
    }


def _batch_summary_row(
    batch_index: int,
    batch_start: pd.Timestamp,
    batch_end: pd.Timestamp,
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "batch_start": batch_start.strftime("%Y-%m-%d"),
        "batch_end": batch_end.strftime("%Y-%m-%d"),
        "status": status,
        "panel_rows": int(len(panel)),
        "daily_rows": int(len(daily)),
        "persistence_passed_rows": _true_count(panel, "persistence_passed"),
        "eligible_for_pool_selection_rows": _true_count(panel, "eligible_for_pool_selection"),
        "eligible_signal_dates": int(panel[panel.get("eligible_for_pool_selection", pd.Series(dtype=str)).astype(str).str.lower().eq("true")]["date"].nunique()) if not panel.empty else 0,
        "error": error,
    }


def _write_progress(
    output: Path,
    panel_parts: list[pd.DataFrame],
    daily_parts: list[pd.DataFrame],
    batch_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    panel = pd.concat(panel_parts, ignore_index=True, sort=False) if panel_parts else _empty_panel()
    daily = pd.concat(daily_parts, ignore_index=True, sort=False) if daily_parts else _empty_daily()
    panel.to_csv(output / "pool2_batched_confirmation_panel.partial.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "pool2_daily_vote_status.partial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(batch_rows).to_csv(output / "pool2_batch_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows, columns=["batch_index", "batch_start", "batch_end", "error"]).to_csv(
        output / "failed.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _gate_breakdown(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in (
        "base_pool_passed",
        "benchmark_margin_passed",
        "momentum_quality_passed",
        "persistence_passed",
        "candidate_support_without_persistence",
        "eligible_for_pool_selection",
    ):
        rows.append(
            {
                "gate": column,
                "rows_true": _true_count(panel, column),
                "rows_false": _false_count(panel, column),
                "rows_missing": int(panel[column].isna().sum()) if column in panel.columns else int(len(panel)),
                "status": "has_true_rows" if _true_count(panel, column) else "zero_true_rows",
            }
        )
    return pd.DataFrame(rows)


def _combined_readiness_blockers(
    pool1_status_output: str | Path,
    pool2_panel: pd.DataFrame,
    eligible: pd.DataFrame,
    failed_rows: list[dict[str, Any]],
    batches: list[list[pd.Timestamp]],
    max_batches: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = Path(pool1_status_output)
    if source.exists() and (source / "manifest.json").exists():
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("pool1_blocked_rows", 0)) > 0:
            rows.append(
                {
                    "field_name": "pool1_full_period_state_stream",
                    "blocker": "pool1_remaining_blocked_rows",
                    "affected_period": "2014-11-03..2015-01-27",
                    "detail": f"Pool1 still has {manifest.get('pool1_blocked_rows')} blocked warmup rows.",
                    "formal_ready": False,
                }
            )
    else:
        rows.append(
            {
                "field_name": "pool1_status_manifest",
                "blocker": "pool1_status_manifest_missing",
                "affected_period": "",
                "detail": f"Missing Pool1 status manifest at {source}",
                "formal_ready": False,
            }
        )
    if max_batches is not None:
        rows.append(
            {
                "field_name": "pool2_batch_scope",
                "blocker": "bounded_smoke_not_full_period",
                "affected_period": _batch_period(batches),
                "detail": "Runner was intentionally bounded by max_batches; full period still needs continuation.",
                "formal_ready": False,
            }
        )
    if failed_rows:
        rows.append(
            {
                "field_name": "pool2_batch_execution",
                "blocker": "failed_batches_present",
                "affected_period": _batch_period(batches),
                "detail": f"{len(failed_rows)} Pool2 batches failed; inspect failed.csv.",
                "formal_ready": False,
            }
        )
    if eligible.empty:
        rows.append(
            {
                "field_name": "pool2_persistence_confirmation",
                "blocker": "no_eligible_pool2_rows",
                "affected_period": _panel_period(pool2_panel),
                "detail": "Batched reconstruction completed but still found no eligible_for_pool_selection rows.",
                "formal_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=["field_name", "blocker", "affected_period", "detail", "formal_ready"])


def _true_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].astype(str).str.lower().eq("true").sum())


def _false_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].astype(str).str.lower().eq("false").sum())


def _batch_period(batches: list[list[pd.Timestamp]]) -> str:
    if not batches:
        return ""
    return f"{batches[0][0].strftime('%Y-%m-%d')}..{batches[-1][-1].strftime('%Y-%m-%d')}"


def _panel_period(panel: pd.DataFrame) -> str:
    if panel.empty or "date" not in panel.columns:
        return ""
    return f"{panel['date'].min()}..{panel['date'].max()}"


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "candidate_ticker",
            "candidate_name",
            "score",
            "eligible_for_pool_selection",
            "persistence_passed",
        ]
    )


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=["signal_date", "pool2_vote", "pool2_confirmation_ready", "pool2_blocker"])


def _handoff_text(panel: pd.DataFrame, eligible: pd.DataFrame, blockers: pd.DataFrame, batches: list[list[pd.Timestamp]], max_batches: int | None) -> str:
    return "\n".join(
        [
            "# Pool2 date-batched persistence reconstruction handoff",
            "",
            f"- executed period: {_batch_period(batches)}",
            f"- bounded smoke: {max_batches is not None}",
            f"- panel rows: {len(panel)}",
            f"- persistence_passed rows: {_true_count(panel, 'persistence_passed')}",
            f"- eligible rows: {len(eligible)}",
            f"- eligible signal dates: {eligible['date'].nunique() if not eligible.empty else 0}",
            f"- combined formal target stream ready: {blockers.empty and max_batches is None}",
            "",
            "## Boundaries",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- This reconstructs Pool2 long-range input data only; it does not change the daily formal model.",
            "",
            "## Remaining blockers",
            *(f"- {row.field_name}: {row.blocker} ({row.affected_period})" for row in blockers.itertuples()),
        ]
    ) + "\n"


def _summary_text(manifest: dict[str, Any], blockers: pd.DataFrame) -> str:
    blocker_lines = "\n".join(f"- {row.field_name}: {row.blocker}" for row in blockers.itertuples()) or "- 無"
    return f"""# Pool2 date-batched persistence reconstruction

- status: {manifest['status']}
- executed batches: {manifest['executed_batches']}/{manifest['requested_batches']}
- pool2 panel rows: {manifest['pool2_panel_rows']}
- persistence_passed rows: {manifest['persistence_passed_rows']}
- eligible_for_pool_selection rows: {manifest['eligible_for_pool_selection_rows']}
- eligible signal dates: {manifest['eligible_signal_dates']}
- combined formal target stream ready: {manifest['combined_formal_target_stream_ready']}
- formal_model_changed=false
- trade_decision_changed=false

## Remaining blockers

{blocker_lines}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run date-batched Pool2 persistence reconstruction.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--monthly-anchor-path", default=DEFAULT_MONTHLY_ANCHOR_PATH)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pool1-status-output", default="outputs/long_range_data_completion_continue_checkpointed_20260702")
    args = parser.parse_args(argv)
    output = run_pool2_date_batched_persistence_reconstruction(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        price_cache_dir=args.price_cache_dir,
        monthly_anchor_path=args.monthly_anchor_path,
        price_source_registry=args.price_source_registry,
        config_path=args.config_path,
        pool1_status_output=args.pool1_status_output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
