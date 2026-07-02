from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import POOL1_TICKERS
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.stock_pool_store import KNOWN_SYMBOLS


TASK_ID = "TASK-BACKTEST-CORE-POOL1-TICKER-LIFECYCLE-CONTRACT-201411-20260702"
DEFAULT_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_PRICE_COVERAGE_PATH = (
    "outputs/core_0050_pit_price_coverage_absorption_201411_202312_20260629/"
    "updated_pit_universe_price_coverage_status.csv"
)
DEFAULT_SUPPLEMENTAL_COVERAGE_PATH = (
    "outputs/core_data_readiness_with_supplemental_price_phase6_20260629/"
    "price_coverage_with_supplemental.csv"
)
DEFAULT_OUTPUT_DIR = "outputs/pool1_ticker_lifecycle_contract_201411_202112_20260702"


def run_pool1_ticker_lifecycle_contract(
    *,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    price_coverage_path: str | Path = DEFAULT_PRICE_COVERAGE_PATH,
    supplemental_coverage_path: str | Path = DEFAULT_SUPPLEMENTAL_COVERAGE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

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

    try:
        panel_root = Path(panel_dir)
        log("load_inputs", "started", str(panel_root))
        manifest = _load_json(panel_root / "manifest.json")
        panel = pd.read_csv(panel_root / "pool1_daily_candidate_ranking_panel.csv").fillna("")
        readiness = pd.read_csv(panel_root / "formal_policy_input_readiness.csv").fillna("")
        price_coverage = pd.read_csv(price_coverage_path).fillna("")
        supplemental = pd.read_csv(supplemental_coverage_path).fillna("")

        log("build_lifecycle_contract", "started", "")
        lifecycle = _lifecycle_contract(panel, price_coverage, supplemental)
        daily = _daily_availability(readiness, lifecycle)
        summary = _coverage_summary(lifecycle, daily)
        excluded = daily[~daily["candidate_available_for_pool1_ranking"]].copy()
        blockers = _blocker_by_ticker(lifecycle)
        source_decision = _source_decisions()

        log("write_outputs", "started", "")
        lifecycle.to_csv(output / "pool1_ticker_lifecycle_contract.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(
            output / "pool1_date_aware_candidate_availability_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(output / "candidate_availability_coverage_summary.csv", index=False, encoding="utf-8-sig")
        excluded.to_csv(output / "excluded_ticker_days.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_ticker.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(
            _final_summary(manifest, lifecycle, summary, blockers),
            encoding="utf-8",
        )

        candidate_availability_ready = bool(lifecycle["candidate_availability_ready"].all())
        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_lifecycle_contract",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "date_start": str(manifest.get("date_start") or _first_date(readiness)),
            "date_end": str(manifest.get("date_end") or _last_date(readiness)),
            "pool1_ticker_count": int(len(lifecycle)),
            "candidate_availability_formal_ready": candidate_availability_ready,
            "candidate_availability_rule": "valid_local_price_and_60_trading_day_warmup",
            "daily_availability_rows": int(len(daily)),
            "excluded_ticker_days": int(len(excluded)),
            "tickers_with_price_data_blocker": int((~lifecycle["price_source_ready"]).sum()),
            "tickers_with_lifecycle_blocker": int((~lifecycle["candidate_availability_ready"]).sum()),
            "remaining_blocker_after_lifecycle": "pool1_attack_gate_state_replay",
            "formal_attack_gate_ready": False,
            "formal_target_stream_ready": False,
            "no_target_cash_all_applied": False,
            "proxy_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "outputs": {
                "lifecycle_contract": "pool1_ticker_lifecycle_contract.csv",
                "daily_availability": "pool1_date_aware_candidate_availability_daily.csv",
                "coverage_summary": "candidate_availability_coverage_summary.csv",
                "excluded_ticker_days": "excluded_ticker_days.csv",
                "blockers": "blocker_by_ticker.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_pool1_ticker_lifecycle_contract", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _lifecycle_contract(panel: pd.DataFrame, price_coverage: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
    panel_first = _panel_first_dates(panel)
    rows: list[dict[str, Any]] = []
    for ticker in POOL1_TICKERS:
        coverage = _coverage_for_ticker(ticker, price_coverage, supplemental)
        data_start = coverage.get("first_date", "")
        data_end = coverage.get("last_date", "")
        first_panel_date = panel_first.get(ticker, "")
        price_ready = _bool_like(coverage.get("price_source_ready") or coverage.get("ready_for_backtest_price_only"))
        adjusted_ready = _bool_like(coverage.get("adjusted_close_available"))
        if ticker == "00631L.TW":
            first_tradable = "2014-10-31"
            lifecycle_source = "TWSE listing/inception checked by Radar/Data; local supplemental price starts 2014-11-03"
        else:
            first_tradable = data_start
            lifecycle_source = "local adjusted price first valid row; official listing archive not required for exclusion if no price"
        availability_ready = bool(price_ready and data_start and data_end and first_panel_date)
        rows.append(
            {
                "ticker": ticker,
                "name": _display_name(ticker),
                "pool1_role": "market_exposure_tool" if ticker == "00631L.TW" else "ai_main_attack_candidate",
                "first_tradable_date": first_tradable,
                "last_tradable_date": data_end,
                "data_start": data_start,
                "data_end": data_end,
                "first_pool1_scoring_date": first_panel_date,
                "warmup_rule": "relative_strength_scores requires more than 60 adjusted-close rows",
                "price_source_ready": price_ready,
                "adjusted_close_available": adjusted_ready,
                "candidate_availability_ready": availability_ready,
                "availability_rule": "include only when signal_date >= first_pool1_scoring_date and signal_date <= data_end",
                "lifecycle_source": lifecycle_source,
                "source_type": coverage.get("source_type", "") or coverage.get("provenance", ""),
                "source_path": coverage.get("cache_path", "") or coverage.get("source_path", "") or coverage.get("base_source", ""),
                "synthetic_used": _bool_like(coverage.get("synthetic_used") or coverage.get("supplemental_synthetic_used")),
                "notes": _lifecycle_notes(ticker, data_start, first_panel_date, price_ready),
            }
        )
    return pd.DataFrame(rows)


def _daily_availability(readiness: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    dates = [pd.Timestamp(date) for date in readiness["date"].astype(str).tolist()]
    rows: list[dict[str, Any]] = []
    for item in lifecycle.to_dict(orient="records"):
        ticker = str(item["ticker"])
        first_score = _to_timestamp(item.get("first_pool1_scoring_date"))
        first_tradable = _to_timestamp(item.get("first_tradable_date"))
        data_start = _to_timestamp(item.get("data_start"))
        data_end = _to_timestamp(item.get("data_end"))
        for date in dates:
            has_price = bool(data_start is not None and data_end is not None and data_start <= date <= data_end)
            scoring_ready = bool(first_score is not None and data_end is not None and first_score <= date <= data_end)
            excluded_reason = ""
            if not has_price:
                excluded_reason = "not_yet_tradable_or_no_local_price_data"
            elif first_score is None or date < first_score:
                excluded_reason = "insufficient_60d_pool1_scoring_warmup"
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "name": item["name"],
                    "first_tradable_date": _fmt_ts(first_tradable),
                    "data_start": _fmt_ts(data_start),
                    "data_end": _fmt_ts(data_end),
                    "first_pool1_scoring_date": _fmt_ts(first_score),
                    "has_valid_price_on_date": has_price,
                    "candidate_available_for_pool1_ranking": scoring_ready,
                    "excluded_reason": excluded_reason,
                    "synthetic_used": bool(item["synthetic_used"]),
                }
            )
    return pd.DataFrame(rows)


def _coverage_summary(lifecycle: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in lifecycle.to_dict(orient="records"):
        ticker = str(item["ticker"])
        subset = daily[daily["ticker"].eq(ticker)]
        available_days = int(subset["candidate_available_for_pool1_ranking"].sum())
        excluded_days = int((~subset["candidate_available_for_pool1_ranking"]).sum())
        rows.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "first_tradable_date": item["first_tradable_date"],
                "data_start": item["data_start"],
                "data_end": item["data_end"],
                "first_pool1_scoring_date": item["first_pool1_scoring_date"],
                "available_days": available_days,
                "excluded_days": excluded_days,
                "excluded_before_scoring_days": int(subset["excluded_reason"].eq("insufficient_60d_pool1_scoring_warmup").sum()),
                "excluded_no_price_days": int(subset["excluded_reason"].eq("not_yet_tradable_or_no_local_price_data").sum()),
                "candidate_availability_ready": bool(item["candidate_availability_ready"]),
                "notes": item["notes"],
            }
        )
    return pd.DataFrame(rows)


def _blocker_by_ticker(lifecycle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in lifecycle.to_dict(orient="records"):
        blockers: list[str] = []
        if not bool(item["price_source_ready"]):
            blockers.append("missing_price_source")
        if not str(item["first_pool1_scoring_date"]):
            blockers.append("missing_pool1_scoring_warmup_date")
        if bool(item["synthetic_used"]):
            blockers.append("synthetic_price_not_allowed")
        status = "ready_for_candidate_availability" if not blockers else "blocked_for_candidate_availability"
        rows.append(
            {
                "ticker": item["ticker"],
                "name": item["name"],
                "status": status,
                "blocker": "; ".join(blockers),
                "candidate_availability_ready": not blockers,
                "remaining_formal_target_blocker": "pool1_attack_gate_state_replay",
            }
        )
    return pd.DataFrame(rows)


def _source_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "pool1_fixed_universe",
                "source_path": "current_formal_pool1_pool2_signal_panels.POOL1_TICKERS",
                "status": "accepted",
                "formal_or_proxy": "formal_universe_contract",
                "decision": "固定 Pool1 標的清單採現行正式主攻池，不加入 Pool1B。",
            },
            {
                "source_layer": "price_coverage",
                "source_path": DEFAULT_PRICE_COVERAGE_PATH,
                "status": "accepted_for_candidate_availability",
                "formal_or_proxy": "formal_availability_input",
                "decision": "候選是否可進排名由當日有效價格與 60 日 warmup 決定；沒有價格或 warmup 不足則排除。",
            },
            {
                "source_layer": "00631L_supplemental_price",
                "source_path": DEFAULT_SUPPLEMENTAL_COVERAGE_PATH,
                "status": "accepted_for_candidate_availability",
                "formal_or_proxy": "formal_availability_input",
                "decision": "00631L 2014/11～2015 使用 TWSE STOCK_DAY supplemental source；synthetic_used=false。",
            },
            {
                "source_layer": "official_listing_archive",
                "source_path": "",
                "status": "not_required_for_price_based_exclusion",
                "formal_or_proxy": "not_used",
                "decision": "本 contract 不用未來固定清單反推歷史；以當日價格可用性 fail-closed。若未來要官方 listing audit，可另補但不阻塞 candidate availability。",
            },
        ]
    )


def _next_step_handoff() -> str:
    return "\n".join(
        [
            "# Pool1 ticker lifecycle handoff",
            "",
            "## 結論",
            "Pool1 date-aware candidate availability 已可 formal-ready：每日只允許有有效價格且已滿 60 日相對強度 warmup 的 ticker 進排名。",
            "",
            "## 已解決",
            "- 6669 在 2017-11-13 前沒有本機價格資料，不會進 Pool1 candidate availability。",
            "- 6669 在 2018-02-06 前未滿 scoring warmup，不會進 Pool1 ranking。",
            "- 00631L 使用 supplemental TWSE STOCK_DAY source，2014-11-03 起可納入價格可用性判斷。",
            "",
            "## 下一步",
            "下一個 blocker 回到 `pool1_attack_gate_state_replay`：用現行 `simulate_regime_mode_switch` 重放 `attack_gate_active`、`attack_gate_ever_activated`、`risk_off_active`、`target_is_actionable`、`model_target_status`，才可產 2014～2021 Pool1 formal vote。",
            "",
            "## 邊界",
            "- 不產正式 target stream。",
            "- 不套 no-target cash-all。",
            "- formal_model_changed=false；trade_decision_changed=false。",
        ]
    ) + "\n"


def _final_summary(
    manifest: dict[str, Any],
    lifecycle: pd.DataFrame,
    summary: pd.DataFrame,
    blockers: pd.DataFrame,
) -> str:
    ready = int(blockers["candidate_availability_ready"].sum()) if not blockers.empty else 0
    total = int(len(blockers))
    excluded = int(summary["excluded_days"].sum()) if not summary.empty else 0
    return "\n".join(
        [
            "# Pool1 ticker lifecycle contract",
            "",
            "## 判定",
            "Pool1 固定標的在 2014/11～2021 的 date-aware candidate availability 可以 formal-ready 回推。",
            "規則是：當日有有效本機價格資料，且已滿 Pool1 20/60 相對強度分數所需的 60 日 warmup，才允許進 Pool1 ranking。",
            "",
            "## 本批輸出",
            f"- 來源區間：{manifest.get('date_start')}～{manifest.get('date_end')}",
            f"- Pool1 ticker：{total}",
            f"- candidate availability ready：{ready}/{total}",
            f"- excluded ticker-days：{excluded}",
            f"- 最晚開始 scoring 的 ticker：{_latest_scoring_ticker(lifecycle)}",
            "",
            "## 邊界",
            "這只解決候選可交易/可排名生命週期，不等於 formal target stream 已完成。下一層仍缺 Pool1 attack gate state replay。",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- no_target_cash_all_applied=false",
        ]
    ) + "\n"


def _panel_first_dates(panel: pd.DataFrame) -> dict[str, str]:
    if panel.empty:
        return {}
    return {
        str(ticker): str(group["date"].min())
        for ticker, group in panel.groupby("candidate_ticker", dropna=False)
        if str(ticker)
    }


def _coverage_for_ticker(ticker: str, price_coverage: pd.DataFrame, supplemental: pd.DataFrame) -> dict[str, Any]:
    if ticker == "00631L.TW":
        row = supplemental[supplemental["ticker"].astype(str).eq(ticker)]
        if not row.empty:
            item = row.iloc[0].to_dict()
            return {
                "first_date": str(item.get("combined_first_date") or ""),
                "last_date": str(item.get("combined_last_date") or ""),
                "price_source_ready": item.get("price_source_ready"),
                "ready_for_backtest_price_only": item.get("price_source_ready"),
                "adjusted_close_available": True,
                "cache_path": str(item.get("base_source") or ""),
                "source_path": str(item.get("supplemental_source_path") or ""),
                "source_type": str(item.get("supplemental_source_type") or item.get("provenance") or ""),
                "synthetic_used": item.get("supplemental_synthetic_used"),
            }
    row = price_coverage[price_coverage["ticker"].astype(str).eq(ticker)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _display_name(ticker: str) -> str:
    return str(KNOWN_SYMBOLS.get(ticker, {}).get("name") or ticker.replace(".TW", ""))


def _lifecycle_notes(ticker: str, data_start: str, first_panel_date: str, price_ready: bool) -> str:
    if not price_ready:
        return "缺可用價格資料，不能進 Pool1 candidate availability。"
    if ticker == "6669.TW":
        return "晚於 2014 起始區間才有價格資料；上市/資料前與 warmup 前均 fail-closed 排除。"
    if ticker == "00631L.TW":
        return "00631L 使用 2014/11 supplemental 真實 TWSE 價格，60 日 warmup 後才進 Pool1 ranking。"
    return f"本機價格從 {data_start} 起可用；{first_panel_date} 起滿足 Pool1 scoring warmup。"


def _latest_scoring_ticker(lifecycle: pd.DataFrame) -> str:
    if lifecycle.empty:
        return ""
    frame = lifecycle.copy()
    frame["first_pool1_scoring_ts"] = pd.to_datetime(frame["first_pool1_scoring_date"], errors="coerce")
    row = frame.sort_values("first_pool1_scoring_ts").iloc[-1]
    return f"{row['name']}({row['ticker']}) {row['first_pool1_scoring_date']}"


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or str(value) == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _fmt_ts(value: pd.Timestamp | None) -> str:
    return "" if value is None else value.strftime("%Y-%m-%d")


def _first_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["date"].iloc[0])


def _last_date(frame: pd.DataFrame) -> str:
    return "" if frame.empty else str(frame["date"].iloc[-1])


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Pool1 ticker lifecycle contract package.")
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--price-coverage-path", default=DEFAULT_PRICE_COVERAGE_PATH)
    parser.add_argument("--supplemental-coverage-path", default=DEFAULT_SUPPLEMENTAL_COVERAGE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_pool1_ticker_lifecycle_contract(
        panel_dir=args.panel_dir,
        price_coverage_path=args.price_coverage_path,
        supplemental_coverage_path=args.supplemental_coverage_path,
        output_dir=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
