from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MARKET_CAP_COLUMNS = ("free_float_market_cap_twd", "market_cap_twd")
DATE_COLUMNS = ("date", "report_date", "source_date")


def load_market_cap_by_ticker(path: str | Path | None, *, signal_date: str | pd.Timestamp) -> dict[str, float]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    frame = pd.read_csv(source, dtype={"symbol": str, "ticker": str}).fillna("")
    if frame.empty:
        return {}
    signal_ts = pd.Timestamp(signal_date).normalize()
    frame = _filter_not_after_signal_date(frame, signal_ts)
    if frame.empty:
        return {}
    cap_columns = [column for column in MARKET_CAP_COLUMNS if column in frame.columns]
    if not cap_columns:
        return {}
    frame["_ticker"] = frame.apply(_row_ticker, axis=1)
    frame["_market_cap_twd"] = _market_cap_series(frame, cap_columns)
    frame = frame[(frame["_ticker"] != "") & (frame["_market_cap_twd"] > 0)].copy()
    if frame.empty:
        return {}
    if "_source_date" in frame.columns:
        frame = frame.sort_values(["_ticker", "_source_date"])
        frame = frame.groupby("_ticker", as_index=False).tail(1)
    else:
        frame = frame.drop_duplicates("_ticker", keep="last")
    return {str(row["_ticker"]): float(row["_market_cap_twd"]) for _, row in frame.iterrows()}


def _market_cap_series(frame: pd.DataFrame, cap_columns: list[str]) -> pd.Series:
    result = pd.Series(0.0, index=frame.index)
    for column in cap_columns:
        values = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
        result = result.where(result > 0, values)
    return result


def discover_market_cap_sources(
    *,
    explicit_path: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
) -> list[Path]:
    sources: list[Path] = []
    if explicit_path:
        sources.append(Path(explicit_path))
    if radar_data_dir:
        data_dir = Path(radar_data_dir)
        sources.extend(
            [
                data_dir / "market_cap.latest.csv",
                data_dir / "market_caps.latest.csv",
                data_dir / "stock_metrics.refreshed.csv",
                data_dir / "formal_radar_candidates.latest.csv",
            ]
        )
    deduped: list[Path] = []
    seen = set()
    for source in sources:
        key = str(source)
        if key in seen:
            continue
        deduped.append(source)
        seen.add(key)
    return deduped


def load_first_available_market_caps(
    *,
    signal_date: str | pd.Timestamp,
    explicit_path: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
) -> tuple[dict[str, float], str]:
    for source in discover_market_cap_sources(explicit_path=explicit_path, radar_data_dir=radar_data_dir):
        caps = load_market_cap_by_ticker(source, signal_date=signal_date)
        if caps:
            return caps, str(source)
    return {}, ""


def _filter_not_after_signal_date(frame: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    date_column = next((column for column in DATE_COLUMNS if column in frame.columns), "")
    if not date_column:
        return frame
    source_dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    filtered = frame.loc[source_dates.notna() & (source_dates <= signal_ts)].copy()
    filtered["_source_date"] = source_dates.loc[filtered.index]
    return filtered


def _row_ticker(row: pd.Series) -> str:
    raw_ticker = str(row.get("ticker") or "").strip().upper()
    if raw_ticker:
        if "." in raw_ticker:
            return raw_ticker
        return f"{raw_ticker}.TW"
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return ""
    suffix = str(row.get("exchange") or row.get("suffix") or "TW").strip().upper() or "TW"
    if suffix.startswith("."):
        suffix = suffix[1:]
    return f"{symbol}.{suffix}"
