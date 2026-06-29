from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv


DEFAULT_PRICE_SOURCE_REGISTRY = "data/price_source_registry.csv"


def load_price_source_registry(path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY) -> pd.DataFrame:
    registry_path = Path(path)
    if not registry_path.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "source_id",
                "source_path",
                "source_type",
                "first_date",
                "last_date",
                "price_source_ready",
                "strategy_ready",
                "synthetic_used",
                "provenance",
                "notes",
            ]
        )
    frame = pd.read_csv(registry_path).fillna("")
    required = {"ticker", "source_path", "source_type", "price_source_ready", "strategy_ready", "synthetic_used"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"price source registry missing columns: {', '.join(sorted(missing))}")
    return frame


def load_supplemental_price_source(
    ticker: str,
    *,
    registry_path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    source_type: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = load_price_source_registry(registry_path)
    normalized_ticker = ticker if ticker.endswith(".TW") else f"{ticker}.TW"
    candidates = registry[registry["ticker"].astype(str).eq(normalized_ticker)]
    if source_type:
        candidates = candidates[candidates["source_type"].astype(str).eq(source_type)]
    candidates = candidates[candidates["price_source_ready"].map(_as_bool)]
    if candidates.empty:
        raise FileNotFoundError(f"No ready supplemental price source for {normalized_ticker}")
    row = candidates.sort_values(["first_date", "last_date"]).iloc[0].to_dict()
    source_path = Path(str(row["source_path"]))
    if not source_path.exists():
        raise FileNotFoundError(f"Supplemental price source file missing: {source_path}")
    frame = load_price_csv(source_path)
    return frame, row


def merge_supplemental_price_source(
    base: pd.DataFrame,
    ticker: str,
    *,
    registry_path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    source_type: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    supplemental, provenance = load_supplemental_price_source(ticker, registry_path=registry_path, source_type=source_type)
    merged = pd.concat([supplemental, base]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged, provenance


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
