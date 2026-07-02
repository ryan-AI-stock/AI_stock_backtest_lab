from __future__ import annotations

import argparse
import json
import urllib.error
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
    payload = _read_json_with_redirects(request)
    return parse_twse_t86_payload(payload, signal_date=signal_date)


def _read_json_with_redirects(request: urllib.request.Request, *, timeout: int = 30, max_redirects: int = 5) -> dict[str, Any]:
    current = request
    for _ in range(max_redirects + 1):
        try:
            with urllib.request.urlopen(current, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            target = urllib.parse.urljoin(current.full_url, location)
            current = urllib.request.Request(target, headers=dict(current.header_items()))
    raise RuntimeError(f"TWSE institutional flow fetch exceeded {max_redirects} redirects")


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


def refresh_twse_institutional_flows(
    *,
    signal_date: str,
    output: str | Path,
    status_json: str | Path | None = None,
    fail_on_empty: bool = False,
) -> dict[str, Any]:
    frame = fetch_twse_institutional_flows(signal_date)
    write_twse_institutional_flows(frame, output)
    status = {
        "requested_signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        "actual_flow_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d") if not frame.empty else "",
        "flow_data_status": "ready" if not frame.empty else "empty_no_rows",
        "row_count": int(len(frame)),
        "output": str(output),
        "active_in_trade_decision": False,
        "boundary": "optional_report_only_risk_factor_source",
        "warning": "" if not frame.empty else f"No TWSE institutional flow rows for {pd.Timestamp(signal_date).strftime('%Y-%m-%d')}; wrote header-only CSV and continued.",
    }
    if status_json:
        path = Path(status_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if fail_on_empty and frame.empty:
        raise SystemExit(status["warning"])
    return status


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
    parser.add_argument("--status-json", default="")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args()

    status = refresh_twse_institutional_flows(
        signal_date=args.signal_date,
        output=args.output,
        status_json=args.status_json or None,
        fail_on_empty=args.fail_on_empty,
    )
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
