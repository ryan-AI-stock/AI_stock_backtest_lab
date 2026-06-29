from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtest_lab.frozen_market_data import roc_date_to_timestamp, twse_float
from backtest_lab.tw50_pit_backfill import ETF_00631L_201411_TRADING_EVIDENCE


TASK_ID = "TASK-BACKTEST-CORE-00631L-PRICE-BACKFILL-201411-201512-PHASE4-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_00631l_price_backfill_201411_201512_phase4_20260629"
TWSE_STOCK_DAY_ENDPOINT = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TWSE_USER_AGENT = "AI_stock_backtest_lab/1.0"
TWSE_MONTHLY_FETCHER = Callable[[str, pd.Period], dict[str, Any]]


def run_twse_stock_day_backfill(
    *,
    ticker: str = "00631L",
    start_month: str = "2014-11",
    end_month: str = "2015-12",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fetcher: TWSE_MONTHLY_FETCHER | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "twse_stock_day_raw").mkdir(exist_ok=True)
    run_log: list[dict[str, str]] = []
    completed_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    normalized_frames: list[pd.DataFrame] = []
    fetch_month = fetcher or fetch_twse_stock_day_month

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    months = list(pd.period_range(start_month, end_month, freq="M"))
    log("initialize", "started", f"{ticker} {start_month} to {end_month}")
    for month in months:
        step = f"fetch_{ticker}_{month}"
        log(step, "started", "")
        try:
            payload = fetch_month(ticker, month)
            raw_path = output / "twse_stock_day_raw" / f"{ticker}_{month}.json"
            raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            frame = normalize_twse_stock_day_payload(payload, ticker=ticker, source_month=str(month))
            if frame.empty:
                raise ValueError("TWSE payload contains no normalized trading rows")
            normalized_frames.append(frame)
            completed_rows.append(
                {
                    "ticker": ticker,
                    "month": str(month),
                    "status": "completed",
                    "row_count": int(len(frame)),
                    "first_date": frame["date"].min(),
                    "last_date": frame["date"].max(),
                    "raw_payload_path": str(raw_path),
                }
            )
            log(step, "completed", f"rows={len(frame)}")
        except Exception as exc:  # noqa: BLE001 - write per-month failure and continue batch.
            failed_rows.append({"ticker": ticker, "month": str(month), "status": "failed", "error": str(exc)})
            log(step, "failed", str(exc))

    normalized = _combine_normalized(normalized_frames)
    normalized_path = output / f"{ticker.lower()}_{start_month.replace('-', '')}_{end_month.replace('-', '')}_twse_stock_day_normalized.csv"
    if ticker == "00631L" and start_month == "2014-11" and end_month == "2015-12":
        normalized_path = output / "00631l_201411_201512_twse_stock_day_normalized.csv"
    normalized.to_csv(normalized_path, index=False, encoding="utf-8-sig")

    completed = pd.DataFrame(completed_rows, columns=["ticker", "month", "status", "row_count", "first_date", "last_date", "raw_payload_path"])
    failed = pd.DataFrame(failed_rows, columns=["ticker", "month", "status", "error"])
    completed.to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")

    coverage = _coverage_after_backfill(ticker, months, normalized, completed, failed)
    coverage.to_csv(output / "00631l_price_coverage_after_backfill.csv", index=False, encoding="utf-8-sig")
    manifest = _manifest(ticker, start_month, end_month, months, completed, failed, normalized, output)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_zh(manifest, coverage), encoding="utf-8")
    log("completed", "completed" if failed.empty else "partial_completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def fetch_twse_stock_day_month(ticker: str, month: pd.Period) -> dict[str, Any]:
    stock_no = ticker.split(".")[0]
    query = urllib.parse.urlencode(
        {
            "date": month.to_timestamp().strftime("%Y%m%d"),
            "stockNo": stock_no,
            "response": "json",
        }
    )
    request = urllib.request.Request(
        f"{TWSE_STOCK_DAY_ENDPOINT}?{query}",
        headers={"User-Agent": TWSE_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_twse_stock_day_payload(payload: dict[str, Any], *, ticker: str, source_month: str) -> pd.DataFrame:
    if payload.get("stat") != "OK":
        raise ValueError(f"TWSE STOCK_DAY fetch failed: {payload.get('stat') or 'unknown status'}")
    fields = [str(field) for field in payload.get("fields", [])]
    data = payload.get("data", [])
    index = _field_index(fields)
    rows: list[dict[str, Any]] = []
    for item in data:
        try:
            close = twse_float(str(item[index["close"]]))
            rows.append(
                {
                    "date": roc_date_to_timestamp(str(item[index["date"]])).strftime("%Y-%m-%d"),
                    "ticker": ticker if ticker.endswith(".TW") else f"{ticker}.TW",
                    "open": twse_float(str(item[index["open"]])),
                    "high": twse_float(str(item[index["high"]])),
                    "low": twse_float(str(item[index["low"]])),
                    "close": close,
                    "adj_close": close,
                    "volume": twse_float(str(item[index["volume"]])),
                    "source": "TWSE_STOCK_DAY",
                    "source_month": source_month,
                    "source_type": "official_real_price",
                    "adjustment_policy": "twse_raw_close_as_adj_close_pending_distribution_review",
                }
            )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unable to parse TWSE row: {item}") from exc
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
                "source_month",
                "source_type",
                "adjustment_policy",
            ]
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _field_index(fields: list[str]) -> dict[str, int]:
    mapping = {
        "date": "日期",
        "volume": "成交股數",
        "open": "開盤價",
        "high": "最高價",
        "low": "最低價",
        "close": "收盤價",
    }
    result: dict[str, int] = {}
    for key, field in mapping.items():
        if field not in fields:
            raise ValueError(f"TWSE fields missing {field}: {fields}")
        result[key] = fields.index(field)
    return result


def _combine_normalized(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
                "source_month",
                "source_type",
                "adjustment_policy",
            ]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def _coverage_after_backfill(
    ticker: str,
    months: list[pd.Period],
    normalized: pd.DataFrame,
    completed: pd.DataFrame,
    failed: pd.DataFrame,
) -> pd.DataFrame:
    completed_months = set(completed["month"].astype(str)) if not completed.empty else set()
    failed_months = set(failed["month"].astype(str)) if not failed.empty else set()
    expected_months = {str(month) for month in months}
    missing_months = sorted(expected_months - completed_months)
    row_count = int(len(normalized))
    return pd.DataFrame(
        [
            {
                "ticker": ticker if ticker.endswith(".TW") else f"{ticker}.TW",
                "source": "TWSE_STOCK_DAY",
                "requested_start_month": str(months[0]) if months else "",
                "requested_end_month": str(months[-1]) if months else "",
                "first_date": normalized["date"].min() if not normalized.empty else "",
                "last_date": normalized["date"].max() if not normalized.empty else "",
                "row_count": row_count,
                "completed_month_count": len(completed_months),
                "failed_month_count": len(failed_months),
                "missing_months": ";".join(missing_months),
                "raw_price_ready": row_count > 0 and not missing_months,
                "formal_ready_for_price_only": row_count > 0 and not missing_months,
                "strategy_ready": False,
                "strategy_ready_blocker": "PIT constituents, formal target stream, and adjusted-close/distribution policy are not fully validated.",
                "synthetic_used": False,
            }
        ]
    )


def _manifest(
    ticker: str,
    start_month: str,
    end_month: str,
    months: list[pd.Period],
    completed: pd.DataFrame,
    failed: pd.DataFrame,
    normalized: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    failed_months = failed["month"].astype(str).tolist() if not failed.empty else []
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed" if not failed_months else "partial_completed_with_failed_months",
        "ticker": ticker if ticker.endswith(".TW") else f"{ticker}.TW",
        "source": "TWSE_STOCK_DAY",
        "start_month": start_month,
        "end_month": end_month,
        "month_job_count": len(months),
        "completed_month_count": int(len(completed)),
        "failed_month_count": int(len(failed)),
        "failed_months": failed_months,
        "normalized_row_count": int(len(normalized)),
        "first_date": normalized["date"].min() if not normalized.empty else "",
        "last_date": normalized["date"].max() if not normalized.empty else "",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "synthetic_used": False,
        "0050x2_proxy_used": False,
        "formal_ready_for_price_only": bool(len(normalized) > 0 and not failed_months),
        "strategy_ready": False,
        "strategy_ready_blocker": "PIT constituents, formal target stream, execution ledger, and distribution-adjustment policy remain incomplete.",
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, Any], coverage: pd.DataFrame) -> str:
    coverage_row = coverage.iloc[0].to_dict() if not coverage.empty else {}
    lines = [
        "# 00631L 2014/11-2015 TWSE STOCK_DAY 真實價格回補",
        "",
        "## 結論",
        "",
        f"- 狀態：{manifest['status']}。",
        f"- 完成月份：{manifest['completed_month_count']} / {manifest['month_job_count']}；失敗月份：{manifest['failed_month_count']}。",
        f"- normalized row count：{manifest['normalized_row_count']}；日期範圍：{manifest['first_date']} 到 {manifest['last_date']}。",
        "- 本次使用 TWSE STOCK_DAY 真實資料；沒有使用 synthetic 0050x2，也沒有 forward-fill 缺月。",
        "- 這只代表 price-only 回補狀態，不代表 2014/11-2023/12 策略已可完整回測。",
        "",
        "## 仍然不是完整策略回測的原因",
        "",
        f"- {coverage_row.get('strategy_ready_blocker', manifest['strategy_ready_blocker'])}",
        "",
        "## 來源證據",
        "",
        f"- {ETF_00631L_201411_TRADING_EVIDENCE}",
    ]
    if manifest["failed_months"]:
        lines.extend(["", "## 失敗月份", "", f"- {', '.join(manifest['failed_months'])}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill a small TWSE STOCK_DAY monthly range into normalized daily prices.")
    parser.add_argument("--ticker", default="00631L")
    parser.add_argument("--start-month", default="2014-11")
    parser.add_argument("--end-month", default="2015-12")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_twse_stock_day_backfill(
        ticker=args.ticker,
        start_month=args.start_month,
        end_month=args.end_month,
        output_dir=args.output_dir,
    )
    print(f"TWSE_STOCK_DAY_BACKFILL_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
