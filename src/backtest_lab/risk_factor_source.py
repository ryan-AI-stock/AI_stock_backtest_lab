from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd


DATE_COLUMNS = ("date", "report_date", "source_date")

SOURCE_FILENAMES = {
    "institutional": (
        "institutional_flows.latest.csv",
        "institutional_flow.latest.csv",
        "institutional_flows.refreshed.csv",
        "stock_metrics.refreshed.csv",
    ),
    "margin_short": (
        "margin_short.latest.csv",
        "margin_short_daily.latest.csv",
        "margin_short.refreshed.csv",
        "stock_metrics.refreshed.csv",
    ),
    "borrow_lending": (
        "borrow_lending.latest.csv",
        "securities_lending.latest.csv",
        "short_lending.latest.csv",
        "stock_metrics.refreshed.csv",
    ),
    "day_trading": (
        "day_trading.latest.csv",
        "day_trading_daily.latest.csv",
        "day_trading.refreshed.csv",
        "stock_metrics.refreshed.csv",
    ),
    "sentiment": (
        "sentiment.latest.csv",
        "social_sentiment.latest.csv",
        "sentiment.refreshed.csv",
        "stock_metrics.refreshed.csv",
    ),
}

FACTOR_COLUMN_HINTS = {
    "institutional": {
        "foreign_net_buy_shares",
        "investment_trust_net_buy_shares",
        "dealer_net_buy_shares",
        "foreign_consecutive_sell_days",
        "trust_consecutive_sell_days",
        "foreign_5d",
        "trust_5d",
    },
    "margin_short": {
        "margin_balance_5d_change_pct",
        "margin_balance_20d_change_pct",
        "short_balance_5d_change_pct",
        "margin_overheat_flag",
        "short_lending_pressure_flag",
        "margin_change_5d",
        "margin_change_20d",
    },
    "borrow_lending": {
        "borrow_balance_5d_change_pct",
        "securities_lending_5d_change_pct",
        "short_lending_5d_change_pct",
        "borrow_sell_ratio",
        "securities_lending_sell_ratio",
        "short_lending_ratio",
        "borrow_pressure_flag",
        "short_lending_pressure_flag",
    },
    "day_trading": {
        "day_trading_volume_ratio",
        "day_trading_ratio",
        "day_trading_ratio_5d_avg",
        "day_trading_overheat_flag",
    },
    "sentiment": {
        "sentiment_score",
        "social_sentiment_score",
        "social_heat_score",
        "message_heat_score",
        "sentiment_heat",
        "sentiment_overheat_flag",
    },
}


@dataclass(frozen=True)
class RiskFactorSignal:
    ticker: str
    signal_date: str = ""
    institutional_risk: float = 0.0
    margin_risk: float = 0.0
    borrow_risk: float = 0.0
    day_trading_risk: float = 0.0
    sentiment_risk: float = 0.0
    bullish_flow_score: float = 0.0
    sentiment_score: float = 0.0
    total_risk_score: float = 0.0
    score_adjustment: float = 0.0
    reasons: tuple[str, ...] = ()
    source_dates: tuple[str, ...] = ()
    source_kinds: tuple[str, ...] = ()

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def load_first_available_risk_factors(
    *,
    signal_date: str | pd.Timestamp,
    radar_data_dir: str | Path | None = None,
    institutional_path: str | Path | None = None,
    margin_short_path: str | Path | None = None,
    borrow_lending_path: str | Path | None = None,
    day_trading_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
) -> tuple[dict[str, RiskFactorSignal], dict[str, str]]:
    explicit_paths = {
        "institutional": institutional_path,
        "margin_short": margin_short_path,
        "borrow_lending": borrow_lending_path,
        "day_trading": day_trading_path,
        "sentiment": sentiment_path,
    }
    signals: dict[str, RiskFactorSignal] = {}
    sources: dict[str, str] = {}
    for kind, explicit_path in explicit_paths.items():
        frame, source = load_first_available_factor_frame(
            kind=kind,
            signal_date=signal_date,
            explicit_path=explicit_path,
            radar_data_dir=radar_data_dir,
        )
        if frame.empty:
            continue
        sources[kind] = source
        for _, row in frame.iterrows():
            ticker = _row_ticker(row)
            if not ticker:
                continue
            current = signals.get(ticker) or RiskFactorSignal(ticker=ticker)
            signals[ticker] = _merge_signal(current, _score_factor_row(kind, ticker, row))
    return signals, sources


def load_first_available_factor_frame(
    *,
    kind: str,
    signal_date: str | pd.Timestamp,
    explicit_path: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    for source in discover_factor_sources(kind=kind, explicit_path=explicit_path, radar_data_dir=radar_data_dir):
        frame = load_factor_frame(source, signal_date=signal_date)
        if not frame.empty and _frame_has_factor_columns(kind, frame):
            return frame, str(source)
    return pd.DataFrame(), ""


def discover_factor_sources(
    *,
    kind: str,
    explicit_path: str | Path | None = None,
    radar_data_dir: str | Path | None = None,
) -> list[Path]:
    sources: list[Path] = []
    if explicit_path:
        sources.append(Path(explicit_path))
    if radar_data_dir:
        data_dir = Path(radar_data_dir)
        for filename in SOURCE_FILENAMES.get(kind, ()):
            sources.append(data_dir / filename)
    deduped: list[Path] = []
    seen = set()
    for source in sources:
        key = str(source)
        if key in seen:
            continue
        deduped.append(source)
        seen.add(key)
    return deduped


def load_factor_frame(path: str | Path | None, *, signal_date: str | pd.Timestamp) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source, dtype={"symbol": str, "ticker": str}).fillna("")
    if frame.empty:
        return pd.DataFrame()
    signal_ts = pd.Timestamp(signal_date).normalize()
    frame = _filter_not_after_signal_date(frame, signal_ts)
    if frame.empty:
        return pd.DataFrame()
    frame["_ticker"] = frame.apply(_row_ticker, axis=1)
    frame = frame[frame["_ticker"] != ""].copy()
    if frame.empty:
        return pd.DataFrame()
    if "_source_date" in frame.columns:
        frame = frame.sort_values(["_ticker", "_source_date"])
        return frame.groupby("_ticker", as_index=False).tail(1).reset_index(drop=True)
    return frame.drop_duplicates("_ticker", keep="last").reset_index(drop=True)


def _score_factor_row(kind: str, ticker: str, row: pd.Series) -> RiskFactorSignal:
    if kind == "institutional":
        return _score_institutional_row(ticker, row)
    if kind == "margin_short":
        return _score_margin_short_row(ticker, row)
    if kind == "borrow_lending":
        return _score_borrow_lending_row(ticker, row)
    if kind == "day_trading":
        return _score_day_trading_row(ticker, row)
    if kind == "sentiment":
        return _score_sentiment_row(ticker, row)
    return RiskFactorSignal(ticker=ticker)


def _frame_has_factor_columns(kind: str, frame: pd.DataFrame) -> bool:
    hints = FACTOR_COLUMN_HINTS.get(kind, set())
    return bool(hints & set(frame.columns))


def _score_institutional_row(ticker: str, row: pd.Series) -> RiskFactorSignal:
    foreign_sell_days = int(_number(row.get("foreign_consecutive_sell_days")))
    trust_sell_days = int(_number(row.get("trust_consecutive_sell_days")))
    foreign_net = _number(row.get("foreign_net_buy_shares") or row.get("foreign_5d"))
    trust_net = _number(row.get("investment_trust_net_buy_shares") or row.get("trust_5d"))
    dealer_net = _number(row.get("dealer_net_buy_shares"))
    total_net = foreign_net + trust_net + dealer_net
    risk = 0.0
    bullish = 0.0
    reasons: list[str] = []
    if foreign_sell_days >= 3:
        risk += min(35.0, 18.0 + (foreign_sell_days - 3) * 4.0)
        reasons.append(f"外資連賣{foreign_sell_days}日")
    if trust_sell_days >= 2:
        risk += min(30.0, 16.0 + (trust_sell_days - 2) * 4.0)
        reasons.append(f"投信連賣{trust_sell_days}日")
    if total_net < 0:
        risk += 8.0
        reasons.append("三大法人合計賣超")
    elif total_net > 0:
        bullish += 8.0
    return _signal(
        ticker,
        row,
        institutional_risk=risk,
        bullish_flow_score=bullish,
        reasons=reasons,
        source_kind="institutional",
    )


def _score_margin_short_row(ticker: str, row: pd.Series) -> RiskFactorSignal:
    margin_5d = _number(row.get("margin_balance_5d_change_pct") or row.get("margin_change_5d"))
    margin_20d = _number(row.get("margin_balance_20d_change_pct") or row.get("margin_change_20d"))
    short_5d = _number(row.get("short_balance_5d_change_pct"))
    risk = 0.0
    reasons: list[str] = []
    if _bool_field(row.get("margin_overheat_flag")) or margin_5d >= 12:
        risk += min(30.0, 12.0 + max(margin_5d - 12.0, 0.0) * 0.8)
        reasons.append("融資短線升溫")
    if margin_20d >= 25:
        risk += 10.0
        reasons.append("融資20日升溫")
    if _bool_field(row.get("short_lending_pressure_flag")) or short_5d >= 15:
        risk += 15.0
        reasons.append("空方/借券壓力升溫")
    return _signal(ticker, row, margin_risk=risk, reasons=reasons, source_kind="margin_short")


def _score_borrow_lending_row(ticker: str, row: pd.Series) -> RiskFactorSignal:
    borrow_change = _number(
        row.get("borrow_balance_5d_change_pct")
        or row.get("securities_lending_5d_change_pct")
        or row.get("short_lending_5d_change_pct")
    )
    borrow_ratio = _number(
        row.get("borrow_sell_ratio")
        or row.get("securities_lending_sell_ratio")
        or row.get("short_lending_ratio")
    )
    risk = 0.0
    reasons: list[str] = []
    if _bool_field(row.get("borrow_pressure_flag") or row.get("short_lending_pressure_flag")):
        risk += 20.0
        reasons.append("借券壓力旗標")
    if borrow_change >= 15:
        risk += min(25.0, 10.0 + (borrow_change - 15.0) * 0.6)
        reasons.append("借券餘額快速增加")
    if borrow_ratio >= 30:
        risk += 15.0
        reasons.append("借券賣出比偏高")
    return _signal(ticker, row, borrow_risk=risk, reasons=reasons, source_kind="borrow_lending")


def _score_day_trading_row(ticker: str, row: pd.Series) -> RiskFactorSignal:
    ratio = _number(row.get("day_trading_volume_ratio") or row.get("day_trading_ratio"))
    avg5 = _number(row.get("day_trading_ratio_5d_avg"))
    risk = 0.0
    reasons: list[str] = []
    if _bool_field(row.get("day_trading_overheat_flag")):
        risk += 20.0
        reasons.append("當沖過熱旗標")
    if ratio >= 35:
        risk += min(25.0, 10.0 + (ratio - 35.0) * 0.4)
        reasons.append(f"當沖比{ratio:.1f}%")
    if avg5 >= 30:
        risk += 10.0
        reasons.append("5日當沖比偏高")
    return _signal(ticker, row, day_trading_risk=risk, reasons=reasons, source_kind="day_trading")


def _score_sentiment_row(ticker: str, row: pd.Series) -> RiskFactorSignal:
    sentiment = _number(row.get("sentiment_score") or row.get("social_sentiment_score"))
    heat = _number(row.get("social_heat_score") or row.get("message_heat_score") or row.get("sentiment_heat"))
    risk = 0.0
    reasons: list[str] = []
    if _bool_field(row.get("sentiment_overheat_flag")):
        risk += 20.0
        reasons.append("社群情緒過熱旗標")
    if heat >= 80 and sentiment > 0:
        risk += 15.0
        reasons.append("社群熱度偏高")
    if sentiment <= -0.6:
        risk += 10.0
        reasons.append("社群情緒轉弱")
    return _signal(
        ticker,
        row,
        sentiment_risk=risk,
        sentiment_score=sentiment,
        reasons=reasons,
        source_kind="sentiment",
    )


def _signal(
    ticker: str,
    row: pd.Series,
    *,
    institutional_risk: float = 0.0,
    margin_risk: float = 0.0,
    borrow_risk: float = 0.0,
    day_trading_risk: float = 0.0,
    sentiment_risk: float = 0.0,
    bullish_flow_score: float = 0.0,
    sentiment_score: float = 0.0,
    reasons: list[str] | None = None,
    source_kind: str,
) -> RiskFactorSignal:
    total_risk = institutional_risk + margin_risk + borrow_risk + day_trading_risk + sentiment_risk
    source_date = _source_date_text(row)
    adjustment = (bullish_flow_score * 0.001) - (total_risk * 0.001)
    return RiskFactorSignal(
        ticker=ticker,
        signal_date=source_date,
        institutional_risk=round(institutional_risk, 4),
        margin_risk=round(margin_risk, 4),
        borrow_risk=round(borrow_risk, 4),
        day_trading_risk=round(day_trading_risk, 4),
        sentiment_risk=round(sentiment_risk, 4),
        bullish_flow_score=round(bullish_flow_score, 4),
        sentiment_score=round(sentiment_score, 4),
        total_risk_score=round(total_risk, 4),
        score_adjustment=round(adjustment, 6),
        reasons=tuple(reasons or ()),
        source_dates=(source_date,) if source_date else (),
        source_kinds=(source_kind,),
    )


def _merge_signal(left: RiskFactorSignal, right: RiskFactorSignal) -> RiskFactorSignal:
    total_risk = (
        left.institutional_risk
        + right.institutional_risk
        + left.margin_risk
        + right.margin_risk
        + left.borrow_risk
        + right.borrow_risk
        + left.day_trading_risk
        + right.day_trading_risk
        + left.sentiment_risk
        + right.sentiment_risk
    )
    bullish = left.bullish_flow_score + right.bullish_flow_score
    sentiment = right.sentiment_score if right.sentiment_score else left.sentiment_score
    return replace(
        left,
        signal_date=right.signal_date or left.signal_date,
        institutional_risk=round(left.institutional_risk + right.institutional_risk, 4),
        margin_risk=round(left.margin_risk + right.margin_risk, 4),
        borrow_risk=round(left.borrow_risk + right.borrow_risk, 4),
        day_trading_risk=round(left.day_trading_risk + right.day_trading_risk, 4),
        sentiment_risk=round(left.sentiment_risk + right.sentiment_risk, 4),
        bullish_flow_score=round(bullish, 4),
        sentiment_score=round(sentiment, 4),
        total_risk_score=round(total_risk, 4),
        score_adjustment=round((bullish * 0.001) - (total_risk * 0.001), 6),
        reasons=tuple(item for item in (*left.reasons, *right.reasons) if item),
        source_dates=tuple(dict.fromkeys(item for item in (*left.source_dates, *right.source_dates) if item)),
        source_kinds=tuple(dict.fromkeys(item for item in (*left.source_kinds, *right.source_kinds) if item)),
    )


def _filter_not_after_signal_date(frame: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    date_column = next((column for column in DATE_COLUMNS if column in frame.columns), "")
    if not date_column:
        return frame
    source_dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    filtered = frame.loc[source_dates.notna() & (source_dates <= signal_ts)].copy()
    filtered["_source_date"] = source_dates.loc[filtered.index]
    return filtered


def _row_ticker(row: pd.Series) -> str:
    raw_ticker = str(row.get("ticker") or row.get("_ticker") or "").strip().upper()
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


def _source_date_text(row: pd.Series) -> str:
    value: Any = row.get("_source_date") or row.get("date") or row.get("report_date") or row.get("source_date") or ""
    if value == "":
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def _bool_field(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
