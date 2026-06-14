from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.stock_pool_store import normalize_ticker
from backtest_lab.valuation_source import load_valuation_signals


DATE_COLUMNS = ("source_date", "report_date", "date")
FAIR_VALUE_COLUMNS = ("fair_price", "target_price", "fair_pe", "target_pe")
EPS_COLUMNS = ("eps_estimate_low", "eps_low", "eps_estimate_high", "eps_high")
BUY_PRICE_COLUMNS = ("buy_price", "max_entry_price")


def build_valuation_readiness(
    *,
    valuation_data: str | Path,
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
    min_average_coverage_ratio: float = 0.60,
) -> dict[str, Any]:
    path = Path(valuation_data)
    if not path.exists():
        return _empty_readiness(
            valuation_data=path,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers or [],
            warnings=["valuation_data_missing"],
        )
    frame = pd.read_csv(path, dtype={"ticker": str, "symbol": str}).fillna("")
    expected = [normalize_ticker(ticker) for ticker in (tickers or []) if normalize_ticker(ticker)]
    warnings = _schema_warnings(frame)
    signal_dates = list(pd.bdate_range(start_date, end_date))
    coverage_rows: list[dict[str, Any]] = []
    all_tickers = set(expected)
    for signal_date in signal_dates:
        signals = load_valuation_signals(
            path,
            signal_date=signal_date,
            current_price_by_ticker={ticker: 100.0 for ticker in expected},
        )
        if not expected:
            all_tickers.update(signals)
        covered = sorted(set(signals) & set(expected)) if expected else sorted(signals)
        denominator = len(expected) if expected else max(len(covered), 1)
        coverage_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "covered_ticker_count": len(covered),
                "expected_ticker_count": denominator,
                "coverage_ratio": len(covered) / denominator if denominator else 0.0,
                "covered_tickers": ",".join(covered),
            }
        )
    if not expected:
        expected = sorted(all_tickers)
    average_coverage = (
        float(pd.Series([row["coverage_ratio"] for row in coverage_rows]).mean())
        if coverage_rows
        else 0.0
    )
    source_dates = _source_dates(frame)
    if frame.empty:
        warnings.append("valuation_data_empty")
    if not source_dates:
        warnings.append("source_date_missing_or_unparseable")
    if not any(column in frame.columns for column in BUY_PRICE_COLUMNS):
        warnings.append("buy_price_missing: can score fair value but cannot reproduce analyst entry-zone logic")
    if average_coverage < min_average_coverage_ratio:
        warnings.append(
            f"coverage_below_threshold:{average_coverage:.2%}<{min_average_coverage_ratio:.2%}"
        )
    status = "ready" if not warnings else "partial"
    if any(warning in warnings for warning in ("valuation_data_empty", "source_date_missing_or_unparseable")):
        status = "not_ready"
    return {
        "status": status,
        "valuation_data": str(path),
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(len(frame)),
        "expected_ticker_count": len(expected),
        "average_coverage_ratio": average_coverage,
        "min_average_coverage_ratio": min_average_coverage_ratio,
        "first_source_date": min(source_dates).strftime("%Y-%m-%d") if source_dates else "",
        "last_source_date": max(source_dates).strftime("%Y-%m-%d") if source_dates else "",
        "warnings": warnings,
        "coverage": coverage_rows,
    }


def write_valuation_readiness_outputs(
    readiness: dict[str, Any],
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> None:
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(readiness.get("coverage", [])).to_csv(path, index=False, encoding="utf-8-sig")


def _empty_readiness(
    *,
    valuation_data: Path,
    start_date: str,
    end_date: str,
    tickers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": "not_ready",
        "valuation_data": str(valuation_data),
        "start_date": start_date,
        "end_date": end_date,
        "row_count": 0,
        "expected_ticker_count": len(tickers),
        "average_coverage_ratio": 0.0,
        "min_average_coverage_ratio": 0.0,
        "first_source_date": "",
        "last_source_date": "",
        "warnings": warnings,
        "coverage": [],
    }


def _schema_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if not {"ticker", "symbol"} & set(frame.columns):
        warnings.append("ticker_or_symbol_missing")
    if not set(DATE_COLUMNS) & set(frame.columns):
        warnings.append("source_date_column_missing")
    has_direct_fair_value = bool(set(FAIR_VALUE_COLUMNS) & set(frame.columns))
    has_eps_route = bool(set(EPS_COLUMNS) & set(frame.columns)) and bool({"fair_pe", "target_pe"} & set(frame.columns))
    if not has_direct_fair_value and not has_eps_route:
        warnings.append("fair_value_columns_missing")
    return warnings


def _source_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    for column in DATE_COLUMNS:
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce").dropna()
        return [pd.Timestamp(date).normalize() for date in dates]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate point-in-time valuation data before shadow backtesting.")
    parser.add_argument("--valuation-data", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--tickers", default="", help="Comma-separated tickers expected in the tested pool.")
    parser.add_argument("--min-average-coverage-ratio", type=float, default=0.60)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    readiness = build_valuation_readiness(
        valuation_data=args.valuation_data,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=tickers or None,
        min_average_coverage_ratio=args.min_average_coverage_ratio,
    )
    write_valuation_readiness_outputs(
        readiness,
        output_json=args.output_json or None,
        output_csv=args.output_csv or None,
    )
    print(json.dumps({key: readiness[key] for key in readiness if key != "coverage"}, ensure_ascii=False, indent=2))
    if args.strict and readiness["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
