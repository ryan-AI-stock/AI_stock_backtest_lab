from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_lab.chip_valuation_event_study import (
    DEFAULT_CACHE_DIRS,
    DEFAULT_VALUATION_SOURCE,
    load_merged_cache_prices,
)
from backtest_lab.config import load_config
from backtest_lab.institutional_flow_overlay_shadow import (
    DEFAULT_DAY_TRADING_SOURCE,
    DEFAULT_FLOW_SOURCE,
    DEFAULT_MARGIN_SOURCE,
    load_day_trading,
    load_institutional_flows,
    load_margin_short,
)


DEFAULT_START_DATE = "2024-01-02"
DEFAULT_END_DATE = "2026-05-26"
DEFAULT_GROUP_ID = "group_c_0050_00631l_plus_mega_caps"
DEFAULT_MARKET_PROXY = "0050.TW"
DEFAULT_MAX_LAG_DAYS = 7
DEFAULT_VALUATION_MAX_LAG_DAYS = 180


@dataclass(frozen=True)
class FactorSourceSpec:
    factor_id: str
    source_path: str | None
    required_columns: tuple[str, ...]
    loader: Callable[[str | Path], pd.DataFrame] | None = None
    max_lag_days: int = DEFAULT_MAX_LAG_DAYS
    applies_to_asset_types: tuple[str, ...] = ("stock",)
    source_kind: str = "daily_point_in_time"


def run_factor_input_coverage_validator(
    *,
    config_path: str,
    group_id: str,
    cache_dirs: list[str],
    output_dir: str,
    start_date: str,
    end_date: str,
    market_proxy: str,
    flow_source: str | None,
    margin_source: str | None,
    day_trading_source: str | None,
    valuation_source: str | None,
    max_lag_days: int,
    valuation_max_lag_days: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
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
        pd.DataFrame(run_log).to_csv(output_path / "run_log.csv", index=False, encoding="utf-8-sig")
        _write_text(output_path / "current_step.txt", step)

    log("load_config", "started", config_path)
    config = load_config(config_path)
    group = config.group_by_id(group_id)
    asset_type_by_ticker = {asset.ticker: asset.asset_type for asset in group.assets}
    expected_tickers = sorted(
        ticker for ticker, asset_type in asset_type_by_ticker.items() if asset_type == "stock"
    )
    log("load_config", "completed", f"group={group_id} stocks={len(expected_tickers)}")

    log("load_trading_calendar", "started", ",".join(cache_dirs))
    prices = load_merged_cache_prices([market_proxy], cache_dirs)
    if market_proxy not in prices:
        raise ValueError(f"Missing market proxy price cache for {market_proxy}")
    trading_dates = _trading_dates(prices[market_proxy], start_date=start_date, end_date=end_date)
    if not trading_dates:
        raise ValueError(f"No trading dates for {market_proxy} in {start_date}~{end_date}")
    log("load_trading_calendar", "completed", f"dates={len(trading_dates)}")

    specs = [
        FactorSourceSpec(
            factor_id="institutional_flows",
            source_path=flow_source,
            required_columns=(
                "date",
                "ticker",
                "foreign_net_buy_shares",
                "investment_trust_net_buy_shares",
                "dealer_net_buy_shares",
                "foreign_consecutive_sell_days",
                "trust_consecutive_sell_days",
            ),
            loader=load_institutional_flows,
            max_lag_days=max_lag_days,
        ),
        FactorSourceSpec(
            factor_id="margin_short",
            source_path=margin_source,
            required_columns=(
                "date",
                "ticker",
                "margin_balance_5d_change_pct",
                "margin_balance_20d_change_pct",
                "short_balance_5d_change_pct",
                "short_balance_20d_change_pct",
                "margin_overheat_flag",
                "short_lending_pressure_flag",
            ),
            loader=load_margin_short,
            max_lag_days=max_lag_days,
        ),
        FactorSourceSpec(
            factor_id="day_trading",
            source_path=day_trading_source,
            required_columns=(
                "date",
                "ticker",
                "day_trading_volume_ratio",
                "day_trading_ratio_5d_avg",
                "day_trading_ratio_20d_avg",
                "day_trading_overheat_flag",
            ),
            loader=load_day_trading,
            max_lag_days=max_lag_days,
        ),
        FactorSourceSpec(
            factor_id="valuation",
            source_path=valuation_source,
            required_columns=("source_date", "ticker", "eps_estimate_low", "eps_estimate_high"),
            loader=None,
            max_lag_days=valuation_max_lag_days,
            source_kind="manual_or_point_in_time_snapshot",
        ),
    ]

    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    log("validate_sources", "started", f"sources={len(specs)}")
    for spec in specs:
        summary, gaps = validate_factor_source(
            spec=spec,
            expected_tickers=expected_tickers,
            trading_dates=trading_dates,
            start_date=start_date,
            end_date=end_date,
        )
        summary_rows.append(summary)
        gap_rows.extend(gaps)
        log(
            f"validate_{spec.factor_id}",
            "completed",
            f"status={summary['readiness_status']} fresh={summary['fresh_coverage_ratio']}",
        )
    log("validate_sources", "completed", "")

    summary_frame = pd.DataFrame(summary_rows)
    gap_frame = pd.DataFrame(gap_rows)
    summary_frame.to_csv(output_path / "factor_input_coverage_summary.csv", index=False, encoding="utf-8-sig")
    gap_frame.to_csv(output_path / "factor_input_gap_list.csv", index=False, encoding="utf-8-sig")
    payload = {
        "model": "factor_input_coverage_validator_v1",
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "config_path": str(Path(config_path).resolve()),
        "group_id": group_id,
        "market_proxy": market_proxy,
        "start_date": start_date,
        "end_date": end_date,
        "trading_date_count": len(trading_dates),
        "expected_stock_tickers": expected_tickers,
        "summary": summary_rows,
        "gap_count": len(gap_rows),
    }
    (output_path / "factor_input_coverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(output_path / "factor_input_coverage_report.md", payload, summary_rows, gap_rows)
    _write_text(output_path / "completed.txt", "completed")
    _write_text(output_path / "current_step.txt", "completed")
    log("completed", "completed", str(output_path.resolve()))
    return output_path


def validate_factor_source(
    *,
    spec: FactorSourceSpec,
    expected_tickers: list[str],
    trading_dates: list[pd.Timestamp],
    start_date: str,
    end_date: str,
) -> tuple[dict, list[dict]]:
    expected_pairs = len(expected_tickers) * len(trading_dates)
    source_path = Path(spec.source_path) if spec.source_path else None
    frame, load_error, missing_columns = _load_source_frame(spec)
    if frame.empty:
        summary = _empty_summary(
            spec=spec,
            source_path=source_path,
            expected_pairs=expected_pairs,
            missing_columns=missing_columns,
            load_error=load_error,
        )
        return summary, [_gap_row(spec, ticker="*", reason=summary["blocked_reason"], missing_dates=len(trading_dates))]

    normalized = _normalize_factor_frame(frame)
    date_min = _date_str(normalized["date"].min()) if "date" in normalized.columns else ""
    date_max = _date_str(normalized["date"].max()) if "date" in normalized.columns else ""
    period = normalized[
        (normalized["date"] >= pd.Timestamp(start_date).normalize())
        & (normalized["date"] <= pd.Timestamp(end_date).normalize())
        & (normalized["ticker"].isin(expected_tickers))
    ].copy()
    exact_pairs = int(period.drop_duplicates(["date", "ticker"]).shape[0])
    latest_by_ticker = _date_lookup_by_ticker(normalized, expected_tickers)
    fresh_counts: dict[str, int] = {ticker: 0 for ticker in expected_tickers}
    for ticker in expected_tickers:
        dates = latest_by_ticker.get(ticker, [])
        for trading_date in trading_dates:
            factor_date = latest_not_after(dates, trading_date)
            if factor_date is None:
                continue
            if (trading_date.normalize() - factor_date.normalize()).days <= spec.max_lag_days:
                fresh_counts[ticker] += 1
    fresh_pairs = int(sum(fresh_counts.values()))
    exact_ratio = _ratio(exact_pairs, expected_pairs)
    fresh_ratio = _ratio(fresh_pairs, expected_pairs)
    freshness_status, blocked_reason = _readiness_status(
        fresh_ratio=fresh_ratio,
        source_start=date_min,
        source_end=date_max,
        required_start=start_date,
        required_end=end_date,
        missing_columns=missing_columns,
        load_error=load_error,
    )
    summary = {
        "factor_id": spec.factor_id,
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "source_kind": spec.source_kind,
        "source_path": str(source_path.resolve()) if source_path else "",
        "source_exists": bool(source_path and source_path.exists()),
        "required_start_date": start_date,
        "required_end_date": end_date,
        "source_start_date": date_min,
        "source_end_date": date_max,
        "source_row_count": int(len(normalized)),
        "source_ticker_count": int(normalized["ticker"].nunique()),
        "expected_symbol_date_count": expected_pairs,
        "exact_symbol_date_count": exact_pairs,
        "exact_coverage_ratio": exact_ratio,
        "fresh_symbol_date_count": fresh_pairs,
        "fresh_coverage_ratio": fresh_ratio,
        "max_lag_days": spec.max_lag_days,
        "missing_columns": ";".join(missing_columns),
        "readiness_status": freshness_status,
        "blocked_reason": blocked_reason,
    }
    gaps = [
        _gap_row(
            spec,
            ticker=ticker,
            reason="fresh_coverage_below_threshold",
            missing_dates=len(trading_dates) - fresh_count,
            fresh_dates=fresh_count,
        )
        for ticker, fresh_count in fresh_counts.items()
        if fresh_count < len(trading_dates)
    ]
    return summary, gaps


def latest_not_after(dates: list[pd.Timestamp], target_date: pd.Timestamp) -> pd.Timestamp | None:
    index = bisect.bisect_right(dates, target_date.normalize()) - 1
    return dates[index] if index >= 0 else None


def _load_source_frame(spec: FactorSourceSpec) -> tuple[pd.DataFrame, str, list[str]]:
    source_path = Path(spec.source_path) if spec.source_path else None
    if not source_path:
        return pd.DataFrame(), "source_path_not_configured", list(spec.required_columns)
    if not source_path.exists():
        return pd.DataFrame(), "source_file_missing", list(spec.required_columns)
    raw = pd.read_csv(source_path, dtype={"ticker": str, "symbol": str}, nrows=0)
    missing_columns = sorted(set(spec.required_columns) - set(raw.columns))
    if missing_columns:
        return pd.DataFrame(), "source_schema_missing_columns", missing_columns
    try:
        if spec.loader is not None:
            return spec.loader(source_path), "", []
        return pd.read_csv(source_path, dtype={"ticker": str, "symbol": str}), "", []
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}", missing_columns


def _normalize_factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    date_column = next((column for column in ("date", "source_date", "report_date") if column in normalized.columns), None)
    if date_column is None:
        normalized["date"] = pd.NaT
    else:
        normalized["date"] = pd.to_datetime(normalized[date_column], errors="coerce").dt.normalize()
    if "ticker" not in normalized.columns:
        normalized["ticker"] = normalized.get("symbol", "").map(_ticker_from_symbol)
    else:
        normalized["ticker"] = normalized["ticker"].map(_ticker_from_symbol)
    return normalized.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def _date_lookup_by_ticker(frame: pd.DataFrame, tickers: list[str]) -> dict[str, list[pd.Timestamp]]:
    result: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        dates = frame.loc[frame["ticker"] == ticker, "date"].drop_duplicates().sort_values().tolist()
        result[ticker] = [pd.Timestamp(date).normalize() for date in dates]
    return result


def _trading_dates(frame: pd.DataFrame, *, start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return [pd.Timestamp(date).normalize() for date in frame.loc[(frame.index >= start) & (frame.index <= end)].index]


def _empty_summary(
    *,
    spec: FactorSourceSpec,
    source_path: Path | None,
    expected_pairs: int,
    missing_columns: list[str],
    load_error: str,
) -> dict:
    return {
        "factor_id": spec.factor_id,
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "source_kind": spec.source_kind,
        "source_path": str(source_path.resolve()) if source_path else "",
        "source_exists": bool(source_path and source_path.exists()),
        "required_start_date": "",
        "required_end_date": "",
        "source_start_date": "",
        "source_end_date": "",
        "source_row_count": 0,
        "source_ticker_count": 0,
        "expected_symbol_date_count": expected_pairs,
        "exact_symbol_date_count": 0,
        "exact_coverage_ratio": 0.0,
        "fresh_symbol_date_count": 0,
        "fresh_coverage_ratio": 0.0,
        "max_lag_days": spec.max_lag_days,
        "missing_columns": ";".join(missing_columns),
        "readiness_status": "blocked",
        "blocked_reason": load_error or "source_empty",
    }


def _readiness_status(
    *,
    fresh_ratio: float,
    source_start: str,
    source_end: str,
    required_start: str,
    required_end: str,
    missing_columns: list[str],
    load_error: str,
) -> tuple[str, str]:
    reasons: list[str] = []
    if missing_columns:
        reasons.append("missing_required_columns")
    if load_error:
        reasons.append(load_error)
    if source_start and source_start > required_end:
        reasons.append("source_starts_after_required_end")
    if source_end and source_end < required_end:
        reasons.append("source_ends_before_required_end")
    if fresh_ratio < 0.95:
        reasons.append("fresh_coverage_below_95pct")
    if fresh_ratio >= 0.95 and not missing_columns and not load_error:
        return "ready", ""
    if fresh_ratio > 0:
        return "partial", ";".join(reasons)
    return "blocked", ";".join(reasons) or "no_fresh_point_in_time_rows"


def _gap_row(
    spec: FactorSourceSpec,
    *,
    ticker: str,
    reason: str,
    missing_dates: int,
    fresh_dates: int = 0,
) -> dict:
    return {
        "factor_id": spec.factor_id,
        "ticker": ticker,
        "decision_layer": "data_readiness",
        "active_in_trade_decision": False,
        "missing_trading_dates": int(missing_dates),
        "fresh_trading_dates": int(fresh_dates),
        "reason": reason,
        "suggested_owner": "RADAR/Research",
        "suggested_action": _suggested_action(spec.factor_id),
    }


def _suggested_action(factor_id: str) -> str:
    if factor_id == "valuation":
        return "補 point-in-time EPS/合理價/買點快照，且 source_date 必須不晚於 signal date。"
    if factor_id == "institutional_flows":
        return "補 2024-2026 三大法人逐日買賣超與連續買賣超欄位。"
    if factor_id == "margin_short":
        return "補 2024-2026 融資融券/借券逐日欄位與過熱旗標。"
    if factor_id == "day_trading":
        return "補 2024-2026 當沖比例逐日欄位與過熱旗標。"
    return "補可追溯 point-in-time source rows。"


def _ticker_from_symbol(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(".TW") or text.endswith(".TWO"):
        return text
    if text.isdigit():
        return f"{text}.TW"
    return text


def _date_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _write_report(path: Path, payload: dict, summary_rows: list[dict], gap_rows: list[dict]) -> None:
    lines = [
        "# Factor/Event-study Input Coverage Validator",
        "",
        f"- decision_layer: `{payload['decision_layer']}`",
        f"- active_in_trade_decision: `{payload['active_in_trade_decision']}`",
        f"- group_id: `{payload['group_id']}`",
        f"- period: `{payload['start_date']}` ~ `{payload['end_date']}`",
        f"- trading_date_count: `{payload['trading_date_count']}`",
        "",
        "## Summary",
        "",
        "| factor | status | fresh coverage | exact coverage | source period | blocked reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {factor_id} | {readiness_status} | {fresh_coverage_ratio:.2%} | {exact_coverage_ratio:.2%} | {source_start_date}~{source_end_date} | {blocked_reason} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Data Requests",
            "",
            "這份輸出只量化資料覆蓋率，不改正式模型。若要讓籌碼/融資/當沖/估值進入 formal challenger，需先補齊下列缺口並重跑驗證。",
            "",
        ]
    )
    for factor_id in sorted({str(row["factor_id"]) for row in gap_rows}):
        sample = next(row for row in gap_rows if row["factor_id"] == factor_id)
        lines.append(f"- `{factor_id}`: {sample['suggested_action']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate factor/event-study input coverage without changing formal models.")
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--cache-dirs", default=DEFAULT_CACHE_DIRS)
    parser.add_argument("--output-dir", default="outputs/factor_input_coverage_20260619")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--market-proxy", default=DEFAULT_MARKET_PROXY)
    parser.add_argument("--flow-source", default=DEFAULT_FLOW_SOURCE)
    parser.add_argument("--margin-source", default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--day-trading-source", default=DEFAULT_DAY_TRADING_SOURCE)
    parser.add_argument("--valuation-source", default=DEFAULT_VALUATION_SOURCE)
    parser.add_argument("--max-lag-days", type=int, default=DEFAULT_MAX_LAG_DAYS)
    parser.add_argument("--valuation-max-lag-days", type=int, default=DEFAULT_VALUATION_MAX_LAG_DAYS)
    args = parser.parse_args()
    run_factor_input_coverage_validator(
        config_path=args.config,
        group_id=args.group_id,
        cache_dirs=[item.strip() for item in args.cache_dirs.split(",") if item.strip()],
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        market_proxy=args.market_proxy,
        flow_source=args.flow_source or None,
        margin_source=args.margin_source or None,
        day_trading_source=args.day_trading_source or None,
        valuation_source=args.valuation_source or None,
        max_lag_days=args.max_lag_days,
        valuation_max_lag_days=args.valuation_max_lag_days,
    )


if __name__ == "__main__":
    main()
