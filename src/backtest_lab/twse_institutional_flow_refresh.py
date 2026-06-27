from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


TWSE_T86_ENDPOINT = "https://www.twse.com.tw/rwd/zh/fund/T86"


def fetch_twse_institutional_flows(signal_date: str) -> pd.DataFrame:
    date_key = pd.Timestamp(signal_date).strftime("%Y%m%d")
    query = urllib.parse.urlencode({"date": date_key, "selectType": "ALLBUT0999", "response": "json"})
    request = urllib.request.Request(
        f"{TWSE_T86_ENDPOINT}?{query}",
        headers={"User-Agent": "AI_stock_backtest_lab/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_twse_t86_payload(payload, signal_date=signal_date)


def parse_twse_t86_payload(payload: dict[str, Any], *, signal_date: str) -> pd.DataFrame:
    fields = [str(item) for item in payload.get("fields") or []]
    data = payload.get("data") or []
    if payload.get("stat") not in {"OK", "很抱歉，沒有符合條件的資料!"} and not data:
        raise ValueError(f"TWSE institutional flow fetch failed: {payload.get('stat') or 'unknown status'}")
    if not fields or not data:
        return pd.DataFrame(columns=_output_columns())
    symbol_i = _field_index(fields, "證券代號")
    name_i = _field_index(fields, "證券名稱")
    foreign_i = _field_index(fields, "外陸資買賣超股數(不含外資自營商)", "外資買賣超股數")
    trust_i = _field_index(fields, "投信買賣超股數")
    dealer_i = _field_index(fields, "自營商買賣超股數")
    rows: list[dict[str, Any]] = []
    for raw in data:
        values = list(raw)
        symbol = _cell(values, symbol_i).strip()
        if not symbol or not symbol.isdigit():
            continue
        foreign_net = _number(_cell(values, foreign_i))
        trust_net = _number(_cell(values, trust_i))
        dealer_net = _number(_cell(values, dealer_i))
        rows.append(
            {
                "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "symbol": symbol,
                "ticker": f"{symbol}.TW",
                "name": _cell(values, name_i).strip(),
                "foreign_net_buy_shares": foreign_net,
                "investment_trust_net_buy_shares": trust_net,
                "dealer_net_buy_shares": dealer_net,
                "total_institutional_net_buy_shares": foreign_net + trust_net + dealer_net,
            }
        )
    return pd.DataFrame(rows, columns=_output_columns())


def write_twse_institutional_flows(frame: pd.DataFrame, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _output_columns() -> list[str]:
    return [
        "date",
        "symbol",
        "ticker",
        "name",
        "foreign_net_buy_shares",
        "investment_trust_net_buy_shares",
        "dealer_net_buy_shares",
        "total_institutional_net_buy_shares",
    ]


def _field_index(fields: list[str], *candidates: str) -> int:
    for candidate in candidates:
        if candidate in fields:
            return fields.index(candidate)
    for candidate in candidates:
        compact = candidate.replace(" ", "")
        for index, field in enumerate(fields):
            if compact in field.replace(" ", ""):
                return index
    return -1


def _cell(values: list[Any], index: int) -> str:
    if index < 0 or index >= len(values):
        return ""
    return str(values[index])


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TWSE T86 institutional flows for stock-pool reports.")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = fetch_twse_institutional_flows(args.signal_date)
    if frame.empty:
        raise SystemExit(f"No TWSE institutional flow rows for {args.signal_date}")
    write_twse_institutional_flows(frame, args.output)
    print(json.dumps({"signal_date": args.signal_date, "row_count": len(frame), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
