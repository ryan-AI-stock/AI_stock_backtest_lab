from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATE_COLUMNS = ("source_date", "report_date", "date")


@dataclass(frozen=True)
class ValuationSignal:
    ticker: str
    signal_date: str = ""
    eps_estimate_low: float = 0.0
    eps_estimate_high: float = 0.0
    fair_pe: float = 0.0
    fair_price: float = 0.0
    buy_price: float = 0.0
    safety_margin_pct: float = 0.0
    gate_passed: bool = True
    score_adjustment: float = 0.0
    reason: str = ""
    source: str = ""


def load_valuation_signals(
    path: str | Path | None,
    *,
    signal_date: str | pd.Timestamp,
    current_price_by_ticker: dict[str, float] | None = None,
) -> dict[str, ValuationSignal]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    frame = pd.read_csv(source, dtype={"ticker": str, "symbol": str}).fillna("")
    if frame.empty:
        return {}
    signal_ts = pd.Timestamp(signal_date).normalize()
    frame = _filter_not_after_signal_date(frame, signal_ts)
    if frame.empty:
        return {}
    frame["_ticker"] = frame.apply(_row_ticker, axis=1)
    frame = frame[frame["_ticker"] != ""].copy()
    if frame.empty:
        return {}
    if "_source_date" in frame.columns:
        frame = frame.sort_values(["_ticker", "_source_date"]).groupby("_ticker", as_index=False).tail(1)
    else:
        frame = frame.drop_duplicates("_ticker", keep="last")
    prices = current_price_by_ticker or {}
    return {
        str(row["_ticker"]): _row_to_signal(row, current_price=prices.get(str(row["_ticker"])), source=str(source))
        for _, row in frame.iterrows()
    }


def _filter_not_after_signal_date(frame: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    for column in DATE_COLUMNS:
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce")
        filtered = frame.loc[dates.notna() & (dates.dt.normalize() <= signal_ts)].copy()
        filtered["_source_date"] = dates.loc[filtered.index].dt.strftime("%Y-%m-%d")
        return filtered
    return frame.copy()


def _row_to_signal(row: pd.Series, *, current_price: float | None, source: str) -> ValuationSignal:
    ticker = str(row["_ticker"])
    eps_low = _number(row.get("eps_estimate_low") or row.get("eps_low"))
    eps_high = _number(row.get("eps_estimate_high") or row.get("eps_high") or eps_low)
    fair_pe = _number(row.get("fair_pe") or row.get("target_pe"))
    fair_price = _number(row.get("fair_price") or row.get("target_price"))
    buy_price = _number(row.get("buy_price") or row.get("max_entry_price"))
    if fair_price <= 0 and fair_pe > 0:
        eps_mid = _midpoint(eps_low, eps_high)
        fair_price = eps_mid * fair_pe if eps_mid > 0 else 0.0
    if buy_price <= 0:
        buy_price = fair_price
    current_price_value = float(current_price or 0.0)
    safety_margin = 0.0
    gate_passed = True
    adjustment = 0.0
    reason = "估值資料僅供診斷"
    if current_price_value > 0 and fair_price > 0:
        safety_margin = fair_price / current_price_value - 1
        if buy_price > 0 and current_price_value > buy_price:
            gate_passed = False
            reason = "現價高於合理買點"
            safety_for_adjustment = buy_price / current_price_value - 1
        elif safety_margin < 0:
            gate_passed = False
            reason = "現價高於合理價"
            safety_for_adjustment = safety_margin
        else:
            reason = "估值仍有安全邊際"
            safety_for_adjustment = safety_margin
        adjustment = max(-0.08, min(0.08, safety_for_adjustment * 0.20))
    source_date = str(row.get("_source_date") or row.get("source_date") or "")
    return ValuationSignal(
        ticker=ticker,
        signal_date=source_date,
        eps_estimate_low=eps_low,
        eps_estimate_high=eps_high,
        fair_pe=fair_pe,
        fair_price=fair_price,
        buy_price=buy_price,
        safety_margin_pct=safety_margin,
        gate_passed=gate_passed,
        score_adjustment=adjustment,
        reason=reason,
        source=source,
    )


def _row_ticker(row: pd.Series) -> str:
    ticker = str(row.get("ticker") or "").strip()
    symbol = str(row.get("symbol") or "").strip()
    raw = ticker or symbol
    if not raw:
        return ""
    if raw.endswith(".TW") or raw.endswith(".TWO"):
        return raw
    if raw.isdigit():
        return f"{raw}.TW"
    return raw


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _midpoint(low: float, high: float) -> float:
    if low > 0 and high > 0:
        return (low + high) / 2
    return max(low, high)
