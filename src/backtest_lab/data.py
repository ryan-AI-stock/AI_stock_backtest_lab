from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PRICE_COLUMNS = ("open", "close", "adj_close")


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Price data missing required columns: {', '.join(missing)}")
    normalized.index = pd.to_datetime(normalized.index).normalize()
    return normalized.sort_index()


def load_price_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return normalize_price_frame(frame)


def split_adjusted_dividends(
    prices: pd.DataFrame,
    manual_splits: tuple[dict[str, float | str], ...] = (),
) -> pd.Series:
    if "dividend" not in prices.columns:
        return pd.Series(0.0, index=prices.index)
    dividends = prices["dividend"].fillna(0.0).astype(float).copy()
    for event in manual_splits:
        split_date = pd.Timestamp(str(event["date"]))
        ratio = float(event["ratio"])
        if ratio <= 0:
            raise ValueError(f"Invalid split ratio for {split_date.date()}: {ratio}")
        dividends.loc[dividends.index < split_date] = dividends.loc[dividends.index < split_date] / ratio
    return dividends


def load_theme_map(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path, dtype={"symbol": str})
    theme_by_ticker: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        theme = str(row.get("theme", "")).strip()
        primary = str(row.get("primary", "yes")).strip().lower()
        if not symbol or not theme or primary in {"no", "false", "0"}:
            continue
        ticker = f"{symbol}.TW"
        theme_by_ticker.setdefault(ticker, [])
        if theme not in theme_by_ticker[ticker]:
            theme_by_ticker[ticker].append(theme)
    return {ticker: tuple(themes) for ticker, themes in theme_by_ticker.items()}


def download_yfinance_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_path / "yfinance"))
    prices: dict[str, pd.DataFrame] = {}
    # yfinance end is exclusive; add one day so the configured final date is included.
    yf_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for ticker in tickers:
        csv_path = cache_path / f"{ticker.replace('.', '_')}.csv"
        if csv_path.exists():
            cached = load_price_csv(csv_path)
            if _covers_range(cached, start_date, end_date):
                prices[ticker] = cached
                continue

        raw = yf.download(
            ticker,
            start=start_date,
            end=yf_end,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError(f"No yfinance data returned for {ticker}")
        raw = _single_ticker_columns(raw, ticker)

        frame = pd.DataFrame(
            {
                "date": raw.index,
                "open": raw["Open"].to_numpy(),
                "high": raw["High"].to_numpy(),
                "low": raw["Low"].to_numpy(),
                "close": raw["Close"].to_numpy(),
                "adj_close": raw["Adj Close"].to_numpy(),
                "volume": raw["Volume"].to_numpy(),
                "dividend": raw.get("Dividends", pd.Series(0, index=raw.index)).to_numpy(),
                "stock_split": raw.get("Stock Splits", pd.Series(0, index=raw.index)).to_numpy(),
            }
        )
        frame.to_csv(csv_path, index=False)
        prices[ticker] = normalize_price_frame(frame.set_index("date"))
    return prices


def _covers_range(frame: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if frame.empty:
        return False
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    first = frame.index.min()
    last = frame.index.max()
    # Config ranges are calendar dates, but Taiwan market data only contains
    # trading dates. Accept a small edge gap so Jan 1 holidays/weekends do not
    # force an unnecessary network refresh when the cache already starts on the
    # first available trading day.
    start_gap_days = (first - start).days
    end_gap_days = (end - last).days
    return start_gap_days <= 10 and end_gap_days <= 10


def _single_ticker_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    if "Ticker" in raw.columns.names:
        return raw.xs(ticker, axis=1, level="Ticker")
    return raw.droplevel(-1, axis=1)
