from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.stock_pool_store import symbol_entry


REQUIRED_COLUMNS = {"effective_date", "ticker"}


def load_tw50_constituents_for_date(path: str | Path, signal_date: str | pd.Timestamp) -> list[dict[str, Any]]:
    """Load Taiwan 50 constituents that were effective on signal_date.

    The file must be point-in-time: every row needs an effective_date, and may
    optionally provide end_date. This avoids using current constituents to
    backfill historical tests.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"TW50 constituent file not found: {source}")
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"TW50 constituent file missing columns: {', '.join(sorted(missing))}")

    target = pd.Timestamp(signal_date).normalize()
    effective = pd.to_datetime(frame["effective_date"], errors="coerce")
    if effective.isna().any():
        raise ValueError("TW50 constituent file contains invalid effective_date values.")
    if "end_date" in frame.columns:
        end = pd.to_datetime(frame["end_date"], errors="coerce")
        active = (effective <= target) & (end.isna() | (end >= target))
    else:
        active = effective <= target
        latest_effective = effective[active].max() if active.any() else pd.NaT
        active = effective == latest_effective
    selected = frame[active].copy()
    if selected.empty:
        raise ValueError(f"No TW50 constituents active on {target.strftime('%Y-%m-%d')}")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in selected.iterrows():
        entry = symbol_entry(str(row["ticker"]), source="tw50_constituents")
        if "name" in selected.columns and str(row.get("name") or "").strip():
            entry["name"] = str(row["name"]).strip()
            entry["display"] = f"{entry['name']}({entry['symbol']})"
        if entry["ticker"] in seen:
            continue
        entries.append(entry)
        seen.add(entry["ticker"])
    return entries

