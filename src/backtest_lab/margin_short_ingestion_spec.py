from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.decision_layers import DIAGNOSTIC


REQUIRED_CANONICAL_COLUMNS = (
    "date",
    "ticker",
    "margin_balance",
    "short_balance",
)

OPTIONAL_CANONICAL_COLUMNS = (
    "margin_buy",
    "margin_sell",
    "margin_cash_repay",
    "short_sell",
    "short_cover",
    "short_stock_repay",
    "margin_balance_5d_change_pct",
    "margin_balance_20d_change_pct",
    "short_balance_5d_change_pct",
    "margin_overheat_flag",
    "short_lending_pressure_flag",
    "source_exchange",
    "source_name",
)

RAW_COLUMN_ALIASES = {
    "date": ("date", "report_date", "source_date", "資料日期"),
    "ticker": ("ticker", "symbol", "stock_id", "證券代號", "股票代號"),
    "name": ("name", "stock_name", "證券名稱", "股票名稱"),
    "margin_buy": ("margin_buy", "融資買進", "融資買進股數"),
    "margin_sell": ("margin_sell", "融資賣出", "融資賣出股數"),
    "margin_cash_repay": ("margin_cash_repay", "融資現金償還", "融資償還"),
    "margin_balance": ("margin_balance", "融資餘額", "融資餘額股數"),
    "short_sell": ("short_sell", "融券賣出", "融券賣出股數"),
    "short_cover": ("short_cover", "融券買進", "融券買進股數"),
    "short_stock_repay": ("short_stock_repay", "融券現券償還", "融券償還"),
    "short_balance": ("short_balance", "融券餘額", "融券餘額股數"),
    "source_exchange": ("source_exchange", "exchange", "市場別"),
}


@dataclass(frozen=True)
class MarginShortIngestionSpec:
    schema_version: int = 1
    source_family: str = "TWSE_TPEx_daily_margin_short_balance"
    decision_layer: str = DIAGNOSTIC
    active_in_trade_decision: bool = False
    canonical_output: str = "margin_short.latest.csv"
    required_columns: tuple[str, ...] = REQUIRED_CANONICAL_COLUMNS
    optional_columns: tuple[str, ...] = OPTIONAL_CANONICAL_COLUMNS
    official_source_notes: str = (
        "TWSE/TPEx daily per-stock margin trading and short selling balance data. "
        "Use prior available source date only; do not forward-fill future rows into a signal date."
    )
    formal_promotion_rule: str = (
        "Diagnostic only until a separate challenger backtest proves better risk-adjusted behavior "
        "without degrading frozen baseline objectives."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_margin_short_ingestion_spec() -> MarginShortIngestionSpec:
    return MarginShortIngestionSpec()


def normalize_margin_short_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize TWSE/TPEx-like margin-short columns into the existing diagnostic schema."""
    if frame.empty:
        return pd.DataFrame(columns=[*REQUIRED_CANONICAL_COLUMNS, *OPTIONAL_CANONICAL_COLUMNS])
    normalized = pd.DataFrame()
    for canonical, aliases in RAW_COLUMN_ALIASES.items():
        source = _first_existing_column(frame, aliases)
        if source:
            normalized[canonical] = frame[source]
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError("Missing required margin-short columns: " + ",".join(missing))
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["ticker"] = normalized["ticker"].map(_normalize_ticker)
    normalized = normalized[normalized["date"].notna() & (normalized["ticker"] != "")].copy()
    for column in set(REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS) & set(normalized.columns):
        if column in {"date", "ticker", "name", "source_exchange", "source_name"}:
            continue
        normalized[column] = normalized[column].map(_number)
    normalized = _add_change_columns(normalized)
    normalized["margin_overheat_flag"] = normalized["margin_balance_5d_change_pct"] >= 12.0
    normalized["short_lending_pressure_flag"] = normalized["short_balance_5d_change_pct"] >= 15.0
    output_columns = [
        "date",
        "ticker",
        *[column for column in OPTIONAL_CANONICAL_COLUMNS if column in normalized.columns],
    ]
    return normalized[output_columns].reset_index(drop=True)


def build_margin_short_readiness(frame: pd.DataFrame, *, signal_date: str) -> dict[str, Any]:
    spec = default_margin_short_ingestion_spec()
    normalized = normalize_margin_short_frame(frame)
    signal_ts = pd.Timestamp(signal_date).normalize()
    source_dates = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    future_rows = int((source_dates > signal_ts).sum())
    dated = normalized.loc[source_dates.notna() & (source_dates <= signal_ts)].copy()
    latest_date = "" if dated.empty else str(source_dates.loc[dated.index].max().date())
    return {
        "schema_version": spec.schema_version,
        "source_family": spec.source_family,
        "decision_layer": spec.decision_layer,
        "active_in_trade_decision": spec.active_in_trade_decision,
        "status": "ready" if not normalized.empty and future_rows == 0 else "blocked",
        "signal_date": signal_date,
        "row_count": int(len(normalized)),
        "ticker_count": int(normalized["ticker"].nunique()) if not normalized.empty else 0,
        "latest_source_date": latest_date,
        "future_data_violation_count": future_rows,
        "required_columns": list(spec.required_columns),
        "optional_columns": list(spec.optional_columns),
        "notes": _readiness_notes(normalized=normalized, future_rows=future_rows),
    }


def write_margin_short_spec_outputs(
    *,
    output_dir: str | Path,
    raw_csv: str | Path | None = None,
    signal_date: str = "",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    spec = default_margin_short_ingestion_spec()
    (output_path / "margin_short_ingestion_spec.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if raw_csv:
        frame = pd.read_csv(raw_csv, dtype=str).fillna("")
        normalized = normalize_margin_short_frame(frame)
        normalized.to_csv(output_path / spec.canonical_output, index=False, encoding="utf-8-sig")
        readiness = build_margin_short_readiness(frame, signal_date=signal_date or str(normalized["date"].max()))
    else:
        readiness = {
            "schema_version": spec.schema_version,
            "source_family": spec.source_family,
            "decision_layer": spec.decision_layer,
            "active_in_trade_decision": spec.active_in_trade_decision,
            "status": "spec_only",
            "signal_date": signal_date,
            "row_count": 0,
            "ticker_count": 0,
            "latest_source_date": "",
            "future_data_violation_count": 0,
            "required_columns": list(spec.required_columns),
            "optional_columns": list(spec.optional_columns),
            "notes": ["raw_csv_not_provided"],
        }
    (output_path / "margin_short_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return readiness


def _first_existing_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    for column in aliases:
        if column in frame.columns:
            return column
    return ""


def _normalize_ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        return text
    return f"{text}.TW"


def _number(value: object) -> float:
    text = str(value or "").strip().replace(",", "")
    if text in {"", "-", "--", "nan", "NaN"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _add_change_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["ticker", "date"]).copy()
    if "margin_balance_5d_change_pct" not in frame.columns:
        frame["margin_balance_5d_change_pct"] = _pct_change_by_ticker(frame, "margin_balance", 5)
    if "margin_balance_20d_change_pct" not in frame.columns:
        frame["margin_balance_20d_change_pct"] = _pct_change_by_ticker(frame, "margin_balance", 20)
    if "short_balance_5d_change_pct" not in frame.columns:
        frame["short_balance_5d_change_pct"] = _pct_change_by_ticker(frame, "short_balance", 5)
    return frame


def _pct_change_by_ticker(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    previous = values.groupby(frame["ticker"]).shift(periods)
    change = ((values - previous) / previous.replace(0, pd.NA)) * 100.0
    return change.fillna(0.0).round(4)


def _readiness_notes(*, normalized: pd.DataFrame, future_rows: int) -> list[str]:
    notes: list[str] = []
    if normalized.empty:
        notes.append("no_valid_rows")
    if future_rows:
        notes.append("future_data_violation")
    notes.append("diagnostic_only_not_formal_trade_signal")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Write TWSE/TPEx margin-short ingestion spec and optional readiness.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-csv", default="")
    parser.add_argument("--signal-date", default="")
    args = parser.parse_args()
    readiness = write_margin_short_spec_outputs(
        output_dir=args.output_dir,
        raw_csv=args.raw_csv or None,
        signal_date=args.signal_date,
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
