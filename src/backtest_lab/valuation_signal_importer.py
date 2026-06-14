from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.stock_pool_store import normalize_ticker


CANONICAL_COLUMNS = [
    "source_date",
    "ticker",
    "symbol",
    "name",
    "eps_estimate_low",
    "eps_estimate_high",
    "fair_pe",
    "fair_price",
    "buy_price",
    "valuation_action",
    "source_name",
    "source_url",
    "notes",
]

ALIASES = {
    "source_date": ("source_date", "report_date", "date", "資料日", "日期"),
    "ticker": ("ticker", "股票代號", "代號"),
    "symbol": ("symbol", "股票代號", "代號"),
    "name": ("name", "股票名稱", "名稱", "公司", "公司名稱"),
    "eps_estimate_low": ("eps_estimate_low", "eps_low", "2026_eps_low", "預估eps低", "預估EPS低"),
    "eps_estimate_high": ("eps_estimate_high", "eps_high", "2026_eps_high", "預估eps高", "預估EPS高"),
    "eps_estimate_range": ("eps_estimate_range", "2026_estimate", "2026預估", "預估EPS", "eps_range"),
    "fair_pe": ("fair_pe", "target_pe", "本益比", "合理本益比", "pe"),
    "fair_price": ("fair_price", "target_price", "合理價", "目標價"),
    "buy_price": ("buy_price", "max_entry_price", "買點", "可買價", "回到幾倍買", "entry_price"),
    "valuation_action": ("valuation_action", "entry_status", "買賣狀態", "動作", "是否可買"),
    "source_name": ("source_name", "資料來源", "來源"),
    "source_url": ("source_url", "來源網址", "url"),
    "notes": ("notes", "備註", "說明"),
}


def normalize_valuation_input_frame(
    frame: pd.DataFrame,
    *,
    default_source_date: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source = frame.fillna("")
    for index, row in source.iterrows():
        normalized = _normalize_row(row, default_source_date=default_source_date)
        if normalized is None:
            skipped.append({"row_index": int(index), "reason": "missing_source_date_or_ticker"})
            continue
        rows.append(normalized)
    output = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if not output.empty:
        output = output.sort_values(["source_date", "ticker"]).reset_index(drop=True)
    return output, {
        "input_rows": int(len(frame)),
        "output_rows": int(len(output)),
        "skipped_rows": skipped,
        "columns": CANONICAL_COLUMNS,
    }


def import_valuation_signals(
    *,
    input_path: str | Path,
    output_path: str | Path,
    default_source_date: str = "",
    append: bool = True,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    frame = pd.read_csv(input_file, dtype=str).fillna("")
    normalized, manifest = normalize_valuation_input_frame(frame, default_source_date=default_source_date)
    if append and output_file.exists():
        existing = pd.read_csv(output_file, dtype=str).fillna("")
        combined = pd.concat([existing, normalized], ignore_index=True)
    else:
        combined = normalized
    if not combined.empty:
        combined = _canonicalize_existing_frame(combined)
        combined = combined.drop_duplicates(["source_date", "ticker"], keep="last")
        combined = combined.sort_values(["source_date", "ticker"]).reset_index(drop=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    return {
        **manifest,
        "input_path": str(input_file),
        "output_path": str(output_file),
        "append": append,
        "final_rows": int(len(combined)),
    }


def _normalize_row(row: pd.Series, *, default_source_date: str) -> dict[str, Any] | None:
    source_date = _text(_first(row, "source_date")) or default_source_date
    ticker_text = _text(_first(row, "ticker") or _first(row, "symbol"))
    if not source_date or not ticker_text:
        return None
    try:
        ticker = normalize_ticker(ticker_text)
    except ValueError:
        return None
    symbol = ticker.split(".")[0]
    eps_low = _number(_first(row, "eps_estimate_low"))
    eps_high = _number(_first(row, "eps_estimate_high"))
    if eps_low <= 0 or eps_high <= 0:
        range_low, range_high = _number_range(_first(row, "eps_estimate_range"))
        eps_low = eps_low or range_low
        eps_high = eps_high or range_high
    if eps_high <= 0:
        eps_high = eps_low
    fair_pe = _number(_first(row, "fair_pe"))
    fair_price = _number(_first(row, "fair_price"))
    if fair_price <= 0 and fair_pe > 0 and max(eps_low, eps_high) > 0:
        fair_price = _midpoint(eps_low, eps_high) * fair_pe
    buy_price = _number(_first(row, "buy_price"))
    valuation_action = _normalize_action(_first(row, "valuation_action"))
    if not valuation_action:
        valuation_action = "buy_zone" if buy_price > 0 else "diagnostic"
    return {
        "source_date": pd.Timestamp(source_date).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "symbol": symbol,
        "name": _text(_first(row, "name")),
        "eps_estimate_low": _format_number(eps_low),
        "eps_estimate_high": _format_number(eps_high),
        "fair_pe": _format_number(fair_pe),
        "fair_price": _format_number(fair_price),
        "buy_price": _format_number(buy_price),
        "valuation_action": valuation_action,
        "source_name": _text(_first(row, "source_name")),
        "source_url": _text(_first(row, "source_url")),
        "notes": _text(_first(row, "notes")),
    }


def _canonicalize_existing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[CANONICAL_COLUMNS]


def _first(row: pd.Series, canonical_name: str) -> object:
    for column in ALIASES[canonical_name]:
        if column in row and _text(row[column]):
            return row[column]
    return ""


def _text(value: object) -> str:
    return str(value or "").strip()


def _number(value: object) -> float:
    text = _text(value).replace(",", "")
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _number_range(value: object) -> tuple[float, float]:
    text = _text(value).replace(",", "")
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return 0.0, 0.0
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def _midpoint(low: float, high: float) -> float:
    if low > 0 and high > 0:
        return (low + high) / 2
    return max(low, high)


def _format_number(value: float) -> str:
    if value <= 0:
        return ""
    rounded = round(value, 4)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.4f}".rstrip("0").rstrip(".")


def _normalize_action(value: object) -> str:
    text = _text(value).lower()
    if not text:
        return ""
    if any(token in text for token in ("不能買", "不可買", "no buy", "cannot", "blocked")):
        return "cannot_buy"
    if any(token in text for token in ("可買", "買點", "buy")):
        return "buy_zone"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize manual valuation rows into valuation_signals CSV format.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-date", default="")
    parser.add_argument("--replace", action="store_true", help="Replace output instead of appending/deduplicating.")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()
    manifest = import_valuation_signals(
        input_path=args.input,
        output_path=args.output,
        default_source_date=args.source_date,
        append=not args.replace,
    )
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
