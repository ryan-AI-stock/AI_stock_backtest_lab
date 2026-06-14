from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.stock_pool_observation import _current_close_by_ticker, _load_observation_price_frames
from backtest_lab.stock_pool_store import normalize_ticker
from backtest_lab.valuation_source import ValuationSignal, load_valuation_signals


def build_valuation_signal_snapshot(
    *,
    valuation_data: str | Path,
    signal_date: str,
    tickers: list[str] | None = None,
    current_price_by_ticker: dict[str, float] | None = None,
) -> dict[str, Any]:
    expected = [normalize_ticker(ticker) for ticker in (tickers or []) if normalize_ticker(ticker)]
    signals = load_valuation_signals(
        valuation_data,
        signal_date=signal_date,
        current_price_by_ticker=current_price_by_ticker or {},
    )
    if not expected:
        expected = sorted(signals)
    rows = [_snapshot_row(ticker, signals.get(ticker), (current_price_by_ticker or {}).get(ticker, 0.0)) for ticker in expected]
    covered = [row for row in rows if row["valuation_status"] != "missing"]
    blocked = [row for row in rows if row["gate_passed"] is False]
    passable = [row for row in rows if row["gate_passed"] is True]
    return {
        "valuation_data": str(valuation_data),
        "signal_date": signal_date,
        "expected_ticker_count": len(expected),
        "covered_ticker_count": len(covered),
        "coverage_ratio": len(covered) / len(expected) if expected else 0.0,
        "blocked_count": len(blocked),
        "passable_count": len(passable),
        "rows": rows,
    }


def write_valuation_signal_snapshot_outputs(
    snapshot: dict[str, Any],
    *,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = snapshot.get("rows", [])
    pd.DataFrame(rows).to_csv(output / "valuation_signal_snapshot.csv", index=False, encoding="utf-8-sig")
    (output / "valuation_signal_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(output / "valuation_signal_snapshot.md", snapshot)


def _snapshot_row(ticker: str, signal: ValuationSignal | None, current_price: float) -> dict[str, Any]:
    if signal is None:
        return {
            "ticker": ticker,
            "signal_date": "",
            "current_price": round(float(current_price or 0.0), 4),
            "valuation_action": "",
            "valuation_status": "missing",
            "gate_passed": "",
            "score_adjustment": 0.0,
            "eps_estimate_low": 0.0,
            "eps_estimate_high": 0.0,
            "fair_pe": 0.0,
            "fair_price": 0.0,
            "buy_price": 0.0,
            "safety_margin_pct": 0.0,
            "reason": "沒有估值資料",
        }
    return {
        "ticker": ticker,
        "signal_date": signal.signal_date,
        "current_price": round(float(current_price or 0.0), 4),
        "valuation_action": signal.valuation_action,
        "valuation_status": "covered",
        "gate_passed": bool(signal.gate_passed),
        "score_adjustment": round(signal.score_adjustment, 4),
        "eps_estimate_low": signal.eps_estimate_low,
        "eps_estimate_high": signal.eps_estimate_high,
        "fair_pe": signal.fair_pe,
        "fair_price": signal.fair_price,
        "buy_price": signal.buy_price,
        "safety_margin_pct": round(signal.safety_margin_pct * 100, 4),
        "reason": signal.reason,
    }


def _write_markdown_report(path: Path, snapshot: dict[str, Any]) -> None:
    lines = [
        "# 估值訊號快照",
        "",
        f"- 訊號日：{snapshot['signal_date']}",
        f"- 估值資料：{snapshot['valuation_data']}",
        f"- 覆蓋率：{snapshot['covered_ticker_count']}/{snapshot['expected_ticker_count']} ({snapshot['coverage_ratio']:.2%})",
        f"- 通過估值閘門：{snapshot['passable_count']}；被擋下：{snapshot['blocked_count']}",
        "",
        "這份快照只用來驗證估值資料如何影響候選分數，不代表正式模型已替換。",
        "",
        "| 標的 | 估值動作 | 現價 | 合理價 | 買點 | 安全邊際 | 閘門 | 原因 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in snapshot.get("rows", []):
        gate = _gate_text(row.get("gate_passed"))
        lines.append(
            f"| {row['ticker']} | {row['valuation_action'] or '-'} | {row['current_price']:,.2f} | "
            f"{row['fair_price']:,.2f} | {row['buy_price']:,.2f} | {row['safety_margin_pct']:.2f}% | "
            f"{gate} | {row['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate_text(value: object) -> str:
    if value is True:
        return "通過"
    if value is False:
        return "擋下"
    return "無資料"


def _parse_tickers(value: str) -> list[str]:
    return [normalize_ticker(item.strip()) for item in value.split(",") if normalize_ticker(item.strip())]


def _load_price_frames_cache_first(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    cache_dirs = _candidate_cache_dirs(cache_dir)
    for ticker in tickers:
        cached = _find_best_cached_frame(ticker, cache_dirs=cache_dirs, end_date=end_date)
        if cached is not None:
            prices[ticker] = cached
        else:
            missing.append(ticker)
    if missing:
        downloaded, still_missing = _load_observation_price_frames(
            tickers=missing,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
        prices.update(downloaded)
        missing = still_missing
    return prices, missing


def _candidate_cache_dirs(cache_dir: str | Path) -> list[Path]:
    roots: list[Path] = []
    for raw in (cache_dir, "backtest_cache"):
        path = Path(raw)
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def _find_best_cached_frame(ticker: str, *, cache_dirs: list[Path], end_date: str) -> pd.DataFrame | None:
    file_name = f"{ticker.replace('.', '_')}.csv"
    end_ts = pd.Timestamp(end_date).normalize()
    best: tuple[pd.Timestamp, pd.DataFrame] | None = None
    for directory in cache_dirs:
        candidates = [directory / file_name]
        candidates.extend(directory.rglob(file_name))
        for path in candidates:
            if not path.exists():
                continue
            try:
                frame = load_price_csv(path)
            except Exception:
                continue
            history = frame.loc[frame.index <= end_ts]
            if history.empty:
                continue
            last = history.index.max()
            if best is None or last > best[0]:
                best = (last, frame)
    return best[1] if best else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a point-in-time valuation signal snapshot.")
    parser.add_argument("--valuation-data", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--cache-dir", default="backtest_cache/stock_pool_observations")
    parser.add_argument("--price-start", default="")
    parser.add_argument("--output-dir", default="outputs/valuation_signal_snapshot")
    parser.add_argument("--no-price", action="store_true")
    args = parser.parse_args()
    tickers = _parse_tickers(args.tickers)
    current_prices: dict[str, float] = {}
    if not args.no_price:
        if not tickers:
            tickers = sorted(load_valuation_signals(args.valuation_data, signal_date=args.signal_date).keys())
        price_start = args.price_start or (pd.Timestamp(args.signal_date) - pd.DateOffset(days=21)).strftime("%Y-%m-%d")
        prices, missing = _load_price_frames_cache_first(
            tickers=tickers,
            start_date=price_start,
            end_date=args.signal_date,
            cache_dir=args.cache_dir,
        )
        current_prices = _current_close_by_ticker(prices, args.signal_date)
        if missing:
            print(f"VALUATION_SIGNAL_SNAPSHOT_MISSING_PRICE_TICKERS={','.join(missing)}")
    snapshot = build_valuation_signal_snapshot(
        valuation_data=args.valuation_data,
        signal_date=args.signal_date,
        tickers=tickers,
        current_price_by_ticker=current_prices,
    )
    write_valuation_signal_snapshot_outputs(snapshot, output_dir=args.output_dir)
    print(f"VALUATION_SIGNAL_SNAPSHOT_OUTPUT={Path(args.output_dir).resolve()}")
    print(json.dumps({key: snapshot[key] for key in snapshot if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
