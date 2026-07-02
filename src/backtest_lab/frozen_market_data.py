from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd


TwseFetcher = Callable[[str, str], dict[str, float] | None]


def append_projection_row(
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    projection_date: pd.Timestamp,
) -> pd.DataFrame:
    projected = frame.loc[frame.index <= signal_date].copy()
    source = projected.loc[signal_date].copy()
    for column in ("dividend", "stock_split"):
        if column in source.index:
            source[column] = 0.0
    projected.loc[projection_date] = source
    return projected.sort_index()


def incomplete_tickers(prices_by_ticker: dict[str, pd.DataFrame], signal_date: str) -> list[str]:
    signal_ts = pd.Timestamp(signal_date)
    incomplete: list[str] = []
    for ticker, frame in prices_by_ticker.items():
        if signal_ts not in frame.index:
            incomplete.append(ticker)
            continue
        row = frame.loc[signal_ts]
        if not _has_complete_price_row(row):
            incomplete.append(ticker)
    return sorted(incomplete)


def fill_signal_date_from_twse(
    prices_by_ticker: dict[str, pd.DataFrame],
    signal_date: str,
    tickers: list[str],
    fetcher: TwseFetcher | None = None,
) -> dict[str, pd.DataFrame]:
    filled = dict(prices_by_ticker)
    fetch = fetcher or fetch_twse_stock_day
    for ticker in tickers:
        try:
            row = fetch(ticker, signal_date)
        except (OSError, urllib.error.URLError, TimeoutError):
            row = None
        if row is None:
            continue
        if not _has_complete_price_row(row):
            continue
        frame = filled[ticker].copy()
        frame.loc[pd.Timestamp(signal_date)] = row
        filled[ticker] = frame.sort_index()
    return filled


def write_price_cache(cache_dir: Path, prices_by_ticker: dict[str, pd.DataFrame], tickers: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        frame = prices_by_ticker[ticker].copy()
        frame.index.name = "date"
        frame.reset_index().to_csv(path, index=False)


def fetch_twse_stock_day(ticker: str, signal_date: str) -> dict[str, float] | None:
    stock_no = ticker.split(".")[0]
    query = urllib.parse.urlencode(
        {
            "date": pd.Timestamp(signal_date).strftime("%Y%m%d"),
            "stockNo": stock_no,
            "response": "json",
        }
    )
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "AI_stock_backtest_lab/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("stat") != "OK":
        return None
    target = pd.Timestamp(signal_date)
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    try:
        date_i = fields.index("日期")
        volume_i = fields.index("成交股數")
        open_i = fields.index("開盤價")
        high_i = fields.index("最高價")
        low_i = fields.index("最低價")
        close_i = fields.index("收盤價")
    except ValueError:
        return None
    for item in data:
        if roc_date_to_timestamp(str(item[date_i])) != target:
            continue
        open_price = twse_float(item[open_i])
        high = twse_float(item[high_i])
        low = twse_float(item[low_i])
        close = twse_float(item[close_i])
        if min(open_price, high, low, close) <= 0:
            return None
        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": twse_float(item[volume_i]),
            "dividend": 0.0,
            "stock_split": 0.0,
        }
    return None


def roc_date_to_timestamp(value: str) -> pd.Timestamp:
    year, month, day = [int(part) for part in value.split("/")]
    return pd.Timestamp(year + 1911, month, day)


def twse_float(value: str) -> float:
    cleaned = str(value).replace(",", "").replace("--", "").strip()
    return float(cleaned) if cleaned else 0.0


def _has_complete_price_row(row: object) -> bool:
    for column in ("open", "high", "low", "close", "adj_close"):
        try:
            value = float(row.get(column))  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            return False
        if pd.isna(value) or value <= 0:
            return False
    return True
