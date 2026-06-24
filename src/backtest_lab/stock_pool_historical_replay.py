from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.risk_factor_source import load_first_available_risk_factors
from backtest_lab.stock_pool_observation import (
    _current_close_by_ticker,
    _load_observation_price_frames,
    _observation_pools,
    _observation_price_tickers,
    _price_start_for_pool,
    _resolve_dynamic_observation_pool,
    _top_candidate_rows,
    build_dispatched_stock_pool_observation,
)
from backtest_lab.stock_pool_store import StockPoolStore
from backtest_lab.valuation_source import load_valuation_signals


DEFAULT_OUTPUT_ROOT = "outputs/stock_pool_historical_replay_20260620"
DEFAULT_PERIODS = {
    "2022": ("2022-01-03", "2022-12-30"),
    "2023": ("2023-01-03", "2023-12-29"),
    "2024_2026": ("2024-01-02", "2026-06-18"),
}
FORWARD_HORIZONS = (20, 60, 120)


@dataclass(frozen=True)
class ReplayResult:
    output_dir: Path
    replay_rows: int
    candidate_rows: int
    forward_rows: int
    failed_rows: int


def run_stock_pool_historical_replay(
    *,
    pool_store_path: str | Path = "data/stock_pools.json",
    cache_dir: str | Path = "backtest_cache/stock_pool_triad_v1_corrected",
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    periods: dict[str, tuple[str, str]] | None = None,
    warmup_start: str = "2020-01-01",
    radar_snapshot_dir: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
    market_cap_data: str | Path | None = None,
    institutional_flow_data: str | Path | None = None,
    margin_short_data: str | Path | None = None,
    borrow_lending_data: str | Path | None = None,
    day_trading_data: str | Path | None = None,
    sentiment_data: str | Path | None = None,
    valuation_data: str | Path | None = None,
    tw50_constituents_path: str | Path | None = "data/tw50_constituents.csv",
    radar_top_n: int = 20,
    candidate_limit: int = 3,
    require_exact_signal_date: bool = True,
    cache_only: bool = True,
    max_dates: int | None = None,
    date_stride: int = 1,
    include_forward_metrics: bool = True,
    signal_calendar_ticker: str = "0050.TW,2330.TW",
) -> ReplayResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    current_step = root / "current_step.txt"
    run_log_path = root / "run_log.csv"
    failed_path = root / "failed.csv"
    replay_path = root / "stock_pool_replay_panel.csv"
    candidates_path = root / "stock_pool_replay_top_candidates.csv"
    forward_path = root / "stock_pool_replay_forward_returns.csv"

    _write_run_log_header(run_log_path)
    failed_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []

    period_ranges = periods or DEFAULT_PERIODS
    store = StockPoolStore(pool_store_path)
    pools = _observation_pools(store.list_pools(), operational_only=True)
    signal_calendar_dates = _load_signal_calendar_dates(
        cache_dir=cache_dir,
        ticker=signal_calendar_ticker,
    ) if cache_only else None
    signal_dates = list(
        _iter_signal_dates(
            period_ranges,
            stride=max(1, date_stride),
            calendar_dates=signal_calendar_dates,
        )
    )
    if max_dates is not None:
        signal_dates = signal_dates[:max_dates]

    market_caps: dict[str, float] = {}
    market_cap_source = ""
    price_frame_cache: dict[str, pd.DataFrame] = {}
    try:
        from backtest_lab.market_cap_source import load_first_available_market_caps

        market_caps, market_cap_source = load_first_available_market_caps(
            signal_date=signal_dates[-1][1] if signal_dates else pd.Timestamp.today().strftime("%Y-%m-%d"),
            explicit_path=market_cap_data,
            radar_data_dir=radar_data_dir,
        )
    except Exception as error:  # pragma: no cover - defensive optional source path
        _append_run_log(run_log_path, "warning", "market_cap_source", str(error))

    for index, (period_name, signal_date) in enumerate(signal_dates, start=1):
        date_text = signal_date.strftime("%Y-%m-%d")
        current_step.write_text(f"{index}/{len(signal_dates)} {period_name} {date_text}\n", encoding="utf-8")
        try:
            risk_signals, risk_sources = load_first_available_risk_factors(
                signal_date=date_text,
                radar_data_dir=radar_data_dir,
                institutional_path=institutional_flow_data,
                margin_short_path=margin_short_data,
                borrow_lending_path=borrow_lending_data,
                day_trading_path=day_trading_data,
                sentiment_path=sentiment_data,
            )
        except Exception as error:
            risk_signals = {}
            risk_sources = {"error": str(error)}
        for pool in pools:
            resolved_pool = _resolve_dynamic_observation_pool(
                pool,
                signal_date=date_text,
                radar_snapshot_dir=radar_snapshot_dir,
                radar_data_dir=radar_data_dir,
                radar_top_n=radar_top_n,
                tw50_constituents_path=tw50_constituents_path,
            )
            pool_id = str(resolved_pool.get("pool_id") or "")
            tickers = [symbol["ticker"] for symbol in resolved_pool.get("resolved_symbols", [])]
            base = {
                "period": period_name,
                "requested_signal_date": date_text,
                "pool_id": pool_id,
                "pool_name": resolved_pool.get("name", ""),
                "strategy_preset": resolved_pool.get("strategy_preset", ""),
                "vote_group": resolved_pool.get("vote_group", ""),
            }
            if not tickers:
                row = {
                    **base,
                    "status": "skipped",
                    "reason": "no_resolved_symbols",
                    "selection_layer": "no_selection",
                    "eligible_for_pool_selection": False,
                }
                replay_rows.append(row)
                failed_rows.append(row)
                _append_run_log(run_log_path, "skipped", f"{date_text}:{pool_id}", "no_resolved_symbols")
                continue
            try:
                price_tickers = _observation_price_tickers(resolved_pool, tickers)
                prices, missing_price_tickers = _load_replay_price_frames(
                    tickers=price_tickers,
                    start_date=_price_start_for_pool(resolved_pool, warmup_start),
                    end_date=date_text,
                    cache_dir=cache_dir,
                    cache_only=cache_only,
                    frame_cache=price_frame_cache,
                )
                if include_forward_metrics:
                    forward_prices, forward_missing = _load_replay_price_frames(
                        tickers=price_tickers,
                        start_date=_price_start_for_pool(resolved_pool, warmup_start),
                        end_date=(signal_date + pd.DateOffset(months=8)).strftime("%Y-%m-%d"),
                        cache_dir=cache_dir,
                        cache_only=cache_only,
                        frame_cache=price_frame_cache,
                    )
                else:
                    forward_prices, forward_missing = {}, []
                if not prices:
                    raise ValueError("no_price_frames")
                valuations = load_valuation_signals(
                    valuation_data,
                    signal_date=date_text,
                    current_price_by_ticker=_current_close_by_ticker(prices, date_text),
                )
                observation = build_dispatched_stock_pool_observation(
                    pool=resolved_pool,
                    prices_by_ticker=prices,
                    signal_date=date_text,
                    warmup_start=warmup_start,
                    market_cap_by_ticker=market_caps,
                    risk_signal_by_ticker=risk_signals,
                    valuation_signal_by_ticker=valuations,
                    require_exact_signal_date=require_exact_signal_date,
                )
                replay_row = _observation_replay_row(
                    base=base,
                    observation=observation,
                    missing_price_tickers=missing_price_tickers,
                    forward_missing_price_tickers=forward_missing,
                    risk_sources=risk_sources,
                )
                replay_rows.append(replay_row)
                for candidate in _top_candidate_rows(observation, limit=candidate_limit):
                    candidate_row = {
                        **base,
                        "status": "generated",
                        "signal_date": observation.signal_date,
                        **_candidate_replay_fields(candidate),
                    }
                    candidate_rows.append(candidate_row)
                    if include_forward_metrics:
                        forward_rows.extend(
                            _forward_return_rows(
                                base=candidate_row,
                                prices_by_ticker=forward_prices,
                                signal_date=observation.signal_date,
                            )
                        )
                if observation.top_ticker and not any(
                    row["ticker"] == observation.top_ticker
                    and row["pool_id"] == pool_id
                    and row["requested_signal_date"] == date_text
                    for row in candidate_rows[-candidate_limit:]
                ):
                    if include_forward_metrics:
                        forward_rows.extend(
                            _forward_return_rows(
                                base={
                                    **base,
                                    "status": "generated",
                                    "signal_date": observation.signal_date,
                                    "rank": 1,
                                    "ticker": observation.top_ticker,
                                    "display": observation.top_display,
                                    "selection_layer": observation.selection_layer,
                                    "eligible_for_pool_selection": observation.eligible_for_pool_selection,
                                    "attack_gate_open": observation.attack_gate_open,
                                    "rank_score": observation.rank_score,
                                    "score": observation.top_score,
                                },
                                prices_by_ticker=forward_prices,
                                signal_date=observation.signal_date,
                            )
                        )
                _append_run_log(run_log_path, "generated", f"{date_text}:{pool_id}", observation.selection_layer)
            except Exception as error:
                row = {
                    **base,
                    "status": "failed",
                    "reason": str(error),
                    "selection_layer": "no_selection",
                    "eligible_for_pool_selection": False,
                }
                replay_rows.append(row)
                failed_rows.append(row)
                _append_run_log(run_log_path, "failed", f"{date_text}:{pool_id}", str(error))

    _write_csv(replay_path, replay_rows)
    _write_csv(candidates_path, candidate_rows)
    _write_csv(forward_path, forward_rows)
    _write_csv(failed_path, failed_rows)
    summary = _summary_rows(replay_rows, candidate_rows, forward_rows)
    _write_csv(root / "stock_pool_replay_summary.csv", summary)
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "historical_stock_pool_replay_snapshot_pack_for_solo_safe_gate_experiments",
        "periods": {key: {"start": start, "end": end} for key, (start, end) in period_ranges.items()},
        "warmup_start": warmup_start,
        "candidate_limit": candidate_limit,
        "require_exact_signal_date": require_exact_signal_date,
        "cache_only": cache_only,
        "max_dates": max_dates,
        "date_stride": date_stride,
        "include_forward_metrics": include_forward_metrics,
        "signal_calendar_ticker": signal_calendar_ticker,
        "signal_calendar_source": "cache_price_dates" if signal_calendar_dates is not None else "business_days",
        "pools": [pool.get("pool_id") for pool in pools],
        "market_cap_source": market_cap_source,
        "forward_horizons": list(FORWARD_HORIZONS),
        "outputs": {
            "replay_panel": str(replay_path),
            "top_candidates": str(candidates_path),
            "forward_returns": str(forward_path),
            "summary": str(root / "stock_pool_replay_summary.csv"),
            "failed": str(failed_path),
            "run_log": str(run_log_path),
        },
        "rows": {
            "replay_panel": len(replay_rows),
            "top_candidates": len(candidate_rows),
            "forward_returns": len(forward_rows),
            "failed": len(failed_rows),
        },
        "contract_fields": {
            "replay_panel": [
                "period",
                "requested_signal_date",
                "signal_date",
                "pool_id",
                "top_ticker",
                "selection_layer",
                "eligible_for_pool_selection",
                "attack_gate_open",
                "gate_rule_id",
                "gate_reason",
            ],
            "top_candidates": [
                "period",
                "signal_date",
                "pool_id",
                "rank",
                "ticker",
                "rank_score",
                "selection_layer",
                "eligible_for_pool_selection",
                "attack_gate_open",
                "gate_rule_id",
                "gate_reason",
            ],
            "forward_returns": [
                "period",
                "signal_date",
                "pool_id",
                "ticker",
                "horizon",
                "forward_return",
                "max_drawdown",
                "max_runup",
                "forward_status",
            ],
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "completed.txt").write_text("completed\n", encoding="utf-8")
    current_step.write_text("completed\n", encoding="utf-8")
    return ReplayResult(
        output_dir=root,
        replay_rows=len(replay_rows),
        candidate_rows=len(candidate_rows),
        forward_rows=len(forward_rows),
        failed_rows=len(failed_rows),
    )


def _iter_signal_dates(
    periods: dict[str, tuple[str, str]],
    *,
    stride: int = 1,
    calendar_dates: set[pd.Timestamp] | None = None,
) -> Iterable[tuple[str, pd.Timestamp]]:
    for period_name, (start, end) in periods.items():
        if calendar_dates is None:
            dates = pd.bdate_range(start, end)
        else:
            start_ts = pd.Timestamp(start).normalize()
            end_ts = pd.Timestamp(end).normalize()
            dates = sorted(date for date in calendar_dates if start_ts <= date <= end_ts)
        for offset, signal_date in enumerate(dates):
            if offset % stride == 0:
                yield period_name, signal_date.normalize()


def _load_signal_calendar_dates(*, cache_dir: str | Path, ticker: str) -> set[pd.Timestamp] | None:
    tickers = [item.strip() for item in ticker.split(",") if item.strip()]
    calendars: list[set[pd.Timestamp]] = []
    for item in tickers:
        csv_path = Path(cache_dir) / f"{item.replace('.', '_')}.csv"
        if not csv_path.exists():
            continue
        try:
            frame = load_price_csv(csv_path)
        except Exception:
            continue
        calendars.append({pd.Timestamp(index).normalize() for index in frame.index})
    if not calendars:
        return None
    common_dates = calendars[0]
    for calendar in calendars[1:]:
        common_dates = common_dates & calendar
    return common_dates


def _load_replay_price_frames(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    cache_only: bool,
    frame_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if not cache_only:
        return _load_observation_price_frames(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    cache_path = Path(cache_dir)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frame_cache = frame_cache if frame_cache is not None else {}
    for ticker in tickers:
        csv_path = cache_path / f"{ticker.replace('.', '_')}.csv"
        if not csv_path.exists():
            missing.append(ticker)
            continue
        try:
            if ticker not in frame_cache:
                frame_cache[ticker] = load_price_csv(csv_path)
            frame = frame_cache[ticker]
        except Exception:
            missing.append(ticker)
            continue
        clipped = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
        if clipped.empty:
            missing.append(ticker)
            continue
        prices[ticker] = clipped
    return prices, missing


def _observation_replay_row(
    *,
    base: dict[str, Any],
    observation: Any,
    missing_price_tickers: list[str],
    forward_missing_price_tickers: list[str],
    risk_sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        "status": "generated",
        "signal_date": observation.signal_date,
        "data_end_date": observation.data_end_date,
        "candidate_count": observation.candidate_count,
        "passed_count": observation.passed_count,
        "top_ticker": observation.top_ticker or "",
        "top_display": observation.top_display or "",
        "top_asset_type": observation.top_asset_type or "",
        "score": observation.top_score,
        "rank_score": observation.rank_score,
        "rank": 1 if observation.top_ticker else "",
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
        "missing_price_tickers": ",".join(missing_price_tickers),
        "forward_missing_price_tickers": ",".join(forward_missing_price_tickers),
        "risk_sources": json.dumps(risk_sources, ensure_ascii=False, sort_keys=True),
    }


def _candidate_replay_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": candidate.get("rank"),
        "strength_rank": candidate.get("strength_rank"),
        "ticker": candidate.get("ticker", ""),
        "display": candidate.get("display", ""),
        "asset_type": candidate.get("asset_type", ""),
        "score": candidate.get("score"),
        "rank_score": candidate.get("rank_score"),
        "passed": candidate.get("passed"),
        "base_pool_passed": candidate.get("base_pool_passed"),
        "attack_gate_open": candidate.get("attack_gate_open"),
        "eligible_for_pool_selection": candidate.get("eligible_for_pool_selection"),
        "selection_layer": candidate.get("selection_layer", ""),
        "selection_label": candidate.get("selection_label", ""),
        "gate_rule_id": candidate.get("gate_rule_id", ""),
        "gate_reason": candidate.get("gate_reason", ""),
        "reason": candidate.get("reason", ""),
    }


def _forward_return_rows(
    *,
    base: dict[str, Any],
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str,
) -> list[dict[str, Any]]:
    ticker = str(base.get("ticker") or "")
    if not ticker:
        return []
    frame = prices_by_ticker.get(ticker)
    rows: list[dict[str, Any]] = []
    for horizon in FORWARD_HORIZONS:
        metrics = _forward_metrics(frame, signal_date=signal_date, horizon=horizon)
        rows.append(
            {
                "period": base.get("period", ""),
                "requested_signal_date": base.get("requested_signal_date", ""),
                "signal_date": signal_date,
                "pool_id": base.get("pool_id", ""),
                "pool_name": base.get("pool_name", ""),
                "rank": base.get("rank", ""),
                "ticker": ticker,
                "display": base.get("display", ""),
                "selection_layer": base.get("selection_layer", ""),
                "eligible_for_pool_selection": base.get("eligible_for_pool_selection", ""),
                "attack_gate_open": base.get("attack_gate_open", ""),
                "rank_score": base.get("rank_score", base.get("score", "")),
                "horizon": horizon,
                **metrics,
            }
        )
    return rows


def _forward_metrics(frame: pd.DataFrame | None, *, signal_date: str, horizon: int) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "forward_status": "missing_price_frame",
            "forward_return": "",
            "max_drawdown": "",
            "max_runup": "",
            "start_price": "",
            "end_price": "",
            "end_date": "",
        }
    signal_ts = pd.Timestamp(signal_date).normalize()
    history = frame.loc[frame.index >= signal_ts].dropna(subset=["adj_close"])
    if len(history) <= horizon:
        return {
            "forward_status": "insufficient_forward_window",
            "forward_return": "",
            "max_drawdown": "",
            "max_runup": "",
            "start_price": float(history["adj_close"].iloc[0]) if not history.empty else "",
            "end_price": "",
            "end_date": "",
        }
    window = history.iloc[: horizon + 1]
    prices = pd.to_numeric(window["adj_close"], errors="coerce").dropna()
    if len(prices) <= horizon:
        return {
            "forward_status": "insufficient_clean_prices",
            "forward_return": "",
            "max_drawdown": "",
            "max_runup": "",
            "start_price": "",
            "end_price": "",
            "end_date": "",
        }
    start_price = float(prices.iloc[0])
    end_price = float(prices.iloc[horizon])
    relative = prices / start_price
    running_max = relative.cummax()
    drawdown = relative / running_max - 1
    return {
        "forward_status": "ready",
        "forward_return": round(end_price / start_price - 1, 8),
        "max_drawdown": round(float(drawdown.min()), 8),
        "max_runup": round(float(relative.max() - 1), 8),
        "start_price": round(start_price, 6),
        "end_price": round(end_price, 6),
        "end_date": prices.index[horizon].strftime("%Y-%m-%d"),
    }


def _summary_rows(
    replay_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_frame = pd.DataFrame(replay_rows)
    if not replay_frame.empty:
        grouped = replay_frame.groupby(["period", "pool_id", "selection_layer"], dropna=False).size().reset_index(name="count")
        rows.extend(grouped.to_dict("records"))
    rows.append({"period": "all", "pool_id": "all", "selection_layer": "candidate_rows", "count": len(candidate_rows)})
    rows.append({"period": "all", "pool_id": "all", "selection_layer": "forward_rows", "count": len(forward_rows)})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event", "target", "message"])
        writer.writeheader()


def _append_run_log(path: Path, event: str, target: str, message: str) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event", "target", "message"])
        writer.writerow({"event": event, "target": target, "message": message})


def _parse_periods(values: list[str] | None) -> dict[str, tuple[str, str]]:
    if not values:
        return DEFAULT_PERIODS
    selected: dict[str, tuple[str, str]] = {}
    for value in values:
        if value in DEFAULT_PERIODS:
            selected[value] = DEFAULT_PERIODS[value]
            continue
        name, raw_range = value.split("=", 1)
        start, end = raw_range.split(":", 1)
        selected[name] = (start, end)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical stock-pool replay snapshot pack.")
    parser.add_argument("--pool-store-path", default="data/stock_pools.json")
    parser.add_argument("--cache-dir", default="backtest_cache/stock_pool_triad_v1_corrected")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--period", action="append", help="Known period key or name=start:end. Can repeat.")
    parser.add_argument("--warmup-start", default="2020-01-01")
    parser.add_argument("--radar-snapshot-dir")
    parser.add_argument("--radar-data-dir")
    parser.add_argument("--market-cap-data")
    parser.add_argument("--institutional-flow-data")
    parser.add_argument("--margin-short-data")
    parser.add_argument("--borrow-lending-data")
    parser.add_argument("--day-trading-data")
    parser.add_argument("--sentiment-data")
    parser.add_argument("--valuation-data")
    parser.add_argument("--tw50-constituents-path", default="data/tw50_constituents.csv")
    parser.add_argument("--radar-top-n", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--allow-download", action="store_true", help="Allow yfinance downloads for missing cache.")
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--date-stride", type=int, default=1)
    parser.add_argument("--skip-forward-metrics", action="store_true")
    parser.add_argument("--signal-calendar-ticker", default="0050.TW,2330.TW")
    parser.add_argument("--allow-nearest-signal-date", action="store_true")
    args = parser.parse_args()
    result = run_stock_pool_historical_replay(
        pool_store_path=args.pool_store_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        periods=_parse_periods(args.period),
        warmup_start=args.warmup_start,
        radar_snapshot_dir=args.radar_snapshot_dir,
        radar_data_dir=args.radar_data_dir,
        market_cap_data=args.market_cap_data,
        institutional_flow_data=args.institutional_flow_data,
        margin_short_data=args.margin_short_data,
        borrow_lending_data=args.borrow_lending_data,
        day_trading_data=args.day_trading_data,
        sentiment_data=args.sentiment_data,
        valuation_data=args.valuation_data,
        tw50_constituents_path=args.tw50_constituents_path,
        radar_top_n=args.radar_top_n,
        candidate_limit=args.candidate_limit,
        require_exact_signal_date=not args.allow_nearest_signal_date,
        cache_only=not args.allow_download,
        max_dates=args.max_dates,
        date_stride=args.date_stride,
        include_forward_metrics=not args.skip_forward_metrics,
        signal_calendar_ticker=args.signal_calendar_ticker,
    )
    print(json.dumps({**asdict(result), "output_dir": str(result.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
