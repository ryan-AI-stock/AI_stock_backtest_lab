from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.market_cap_source import load_first_available_market_caps
from backtest_lab.risk_factor_source import RiskFactorSignal, load_first_available_risk_factors


REQUIRED_RISK_KINDS = ("institutional", "margin_short", "borrow_lending", "day_trading", "sentiment")


def build_risk_factor_readiness(
    *,
    signal_date: str,
    radar_data_dir: str | Path | None = None,
    market_cap_data: str | Path | None = None,
    institutional_flow_data: str | Path | None = None,
    margin_short_data: str | Path | None = None,
    borrow_lending_data: str | Path | None = None,
    day_trading_data: str | Path | None = None,
    sentiment_data: str | Path | None = None,
) -> dict[str, Any]:
    market_caps, market_cap_source = load_first_available_market_caps(
        signal_date=signal_date,
        explicit_path=market_cap_data,
        radar_data_dir=radar_data_dir,
    )
    risk_signals, risk_sources = load_first_available_risk_factors(
        signal_date=signal_date,
        radar_data_dir=radar_data_dir,
        institutional_path=institutional_flow_data,
        margin_short_path=margin_short_data,
        borrow_lending_path=borrow_lending_data,
        day_trading_path=day_trading_data,
        sentiment_path=sentiment_data,
    )
    available_kinds = sorted(risk_sources)
    missing_kinds = [kind for kind in REQUIRED_RISK_KINDS if kind not in risk_sources]
    nonzero_risk_count = sum(1 for signal in risk_signals.values() if signal.total_risk_score > 0)
    dated_signal_count = sum(1 for signal in risk_signals.values() if signal.source_dates)
    return {
        "status": "ready" if not missing_kinds and market_caps else "partial",
        "signal_date": signal_date,
        "radar_data_dir": str(radar_data_dir or ""),
        "market_cap_source": market_cap_source,
        "market_cap_count": len(market_caps),
        "risk_factor_sources": risk_sources,
        "risk_factor_count": len(risk_signals),
        "risk_factor_nonzero_count": nonzero_risk_count,
        "risk_factor_dated_count": dated_signal_count,
        "available_risk_kinds": available_kinds,
        "missing_risk_kinds": missing_kinds,
        "notes": _readiness_notes(
            market_cap_count=len(market_caps),
            risk_signals=risk_signals,
            missing_kinds=missing_kinds,
            dated_signal_count=dated_signal_count,
            nonzero_risk_count=nonzero_risk_count,
        ),
        "signals": [_signal_row(signal) for signal in sorted(risk_signals.values(), key=lambda item: item.ticker)],
    }


def write_readiness_outputs(
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
        pd.DataFrame(readiness.get("signals", [])).to_csv(path, index=False, encoding="utf-8-sig")


def _readiness_notes(
    *,
    market_cap_count: int,
    risk_signals: dict[str, RiskFactorSignal],
    missing_kinds: list[str],
    dated_signal_count: int,
    nonzero_risk_count: int,
) -> list[str]:
    notes: list[str] = []
    if market_cap_count == 0:
        notes.append("market_cap_missing")
    if missing_kinds:
        notes.append("missing_risk_kinds:" + ",".join(missing_kinds))
    if risk_signals and dated_signal_count == 0:
        notes.append("risk_factor_rows_have_no_source_date")
    if risk_signals and nonzero_risk_count == 0:
        notes.append("risk_factor_values_all_zero_or_not_flagged")
    return notes


def _signal_row(signal: RiskFactorSignal) -> dict[str, Any]:
    return {
        "ticker": signal.ticker,
        "total_risk_score": signal.total_risk_score,
        "institutional_risk": signal.institutional_risk,
        "margin_risk": signal.margin_risk,
        "borrow_risk": signal.borrow_risk,
        "day_trading_risk": signal.day_trading_risk,
        "sentiment_risk": signal.sentiment_risk,
        "bullish_flow_score": signal.bullish_flow_score,
        "sentiment_score": signal.sentiment_score,
        "score_adjustment": signal.score_adjustment,
        "source_dates": ",".join(signal.source_dates),
        "source_kinds": ",".join(signal.source_kinds),
        "reasons": signal.reason_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate optional risk-factor data sources for stock-pool observation.")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--radar-data-dir", default="")
    parser.add_argument("--market-cap-data", default="")
    parser.add_argument("--institutional-flow-data", default="")
    parser.add_argument("--margin-short-data", default="")
    parser.add_argument("--borrow-lending-data", default="")
    parser.add_argument("--day-trading-data", default="")
    parser.add_argument("--sentiment-data", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    readiness = build_risk_factor_readiness(
        signal_date=args.signal_date,
        radar_data_dir=args.radar_data_dir or None,
        market_cap_data=args.market_cap_data or None,
        institutional_flow_data=args.institutional_flow_data or None,
        margin_short_data=args.margin_short_data or None,
        borrow_lending_data=args.borrow_lending_data or None,
        day_trading_data=args.day_trading_data or None,
        sentiment_data=args.sentiment_data or None,
    )
    write_readiness_outputs(readiness, output_json=args.output_json or None, output_csv=args.output_csv or None)
    print(json.dumps({key: readiness[key] for key in readiness if key != "signals"}, ensure_ascii=False, indent=2))
    if args.strict and readiness["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
