from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.current_formal_pool1_pool2_signal_panels import (
    DEFAULT_PRICE_CACHE_DIR,
    DEFAULT_PRICE_SOURCE_REGISTRY,
    POOL1_TICKERS,
    _load_price_source,
    _load_price_source_registry,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET


TASK_ID = "TASK-BACKTEST-CORE-POOL1-WARMUP-BOOTSTRAP-POLICY-DIAGNOSTIC-20260702"
DEFAULT_LONG_RANGE_DIR = "outputs/long_range_data_completion_continue_checkpointed_20260702"
DEFAULT_LIFECYCLE_DIR = "outputs/pool1_ticker_lifecycle_contract_201411_202112_20260702"
DEFAULT_POOL2_DIR = "outputs/pool2_date_batched_persistence_reconstruction_201411_202112_20260702"
DEFAULT_OUTPUT_DIR = "outputs/pool1_warmup_bootstrap_policy_diagnostic_201411_201501_20260702"
MIN_DYNAMIC_POOL1_UNIVERSE = 7
MIN_POOL1_HISTORY_ROWS = 61


def run_pool1_warmup_bootstrap_policy_diagnostic(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    long_range_dir: str | Path = DEFAULT_LONG_RANGE_DIR,
    lifecycle_dir: str | Path = DEFAULT_LIFECYCLE_DIR,
    pool2_dir: str | Path = DEFAULT_POOL2_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        long_root = Path(long_range_dir)
        lifecycle_root = Path(lifecycle_dir)
        pool2_root = Path(pool2_dir)
        log("load_inputs", "started", f"{long_root}; {lifecycle_root}; {pool2_root}")
        blocked = pd.read_csv(long_root / "remaining_blocked_rows.csv").fillna("")
        lifecycle = pd.read_csv(lifecycle_root / "pool1_ticker_lifecycle_contract.csv").fillna("")
        pool1_replay = _read_optional_csv(long_root / "pool1_full_state_replay_201411_202112.csv")
        pool2_daily = _read_optional_csv(pool2_root / "pool2_daily_vote_status.csv")
        pool2_eligible = _read_optional_csv(pool2_root / "pool2_reconstructed_eligible_rows.csv")
        registry = _load_price_source_registry(price_source_registry)

        log("load_pool1_prices", "started", "")
        prices, price_meta, missing_prices = _load_pool1_prices(price_cache_dir, registry)

        log("build_blocker_breakdown", "started", "")
        row_breakdown = _blocked_row_breakdown(blocked, lifecycle, prices)
        blocker_summary = _blocker_summary(row_breakdown)
        option_table = _policy_options(row_breakdown)
        candidate_warmup = _candidate_warmup_table(row_breakdown)
        combined_readiness = _combined_readiness(row_breakdown, pool1_replay, pool2_daily, pool2_eligible)
        source_decision = _source_decision(price_meta, missing_prices)

        log("write_outputs", "started", str(output))
        row_breakdown.to_csv(output / "pool1_warmup_blocker_breakdown.csv", index=False, encoding="utf-8-sig")
        blocker_summary.to_csv(output / "pool1_warmup_blocker_summary.csv", index=False, encoding="utf-8-sig")
        option_table.to_csv(output / "warmup_policy_options.csv", index=False, encoding="utf-8-sig")
        candidate_warmup.to_csv(output / "diagnostic_bootstrap_candidate_warmup_table.csv", index=False, encoding="utf-8-sig")
        combined_readiness.to_csv(output / "combined_formal_target_stream_warmup_readiness.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "warmup_policy_options.md").write_text(_options_markdown(option_table), encoding="utf-8")
        (output / "next_step_handoff.md").write_text(_handoff_text(combined_readiness), encoding="utf-8")

        recommended = option_table[option_table["recommended"].astype(str).str.lower().eq("true")].iloc[0].to_dict()
        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_diagnostic_recommendation",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "blocked_rows": int(len(row_breakdown)),
            "blocked_period": _period(row_breakdown, "signal_date"),
            "rows_with_zero_scoring_universe": int(row_breakdown["available_universe_count"].eq(0).sum()),
            "rows_with_partial_scoring_universe": int(
                ((row_breakdown["available_universe_count"] > 0) & (row_breakdown["available_universe_count"] < MIN_DYNAMIC_POOL1_UNIVERSE)).sum()
            ),
            "first_formal_candidate_signal_date_after_warmup": str(combined_readiness.iloc[0]["first_pool1_replay_date"]),
            "recommended_policy": recommended.get("option_id", ""),
            "combined_formal_target_stream_full_201411_ready": False,
            "combined_formal_target_stream_ready_from_warmup_start": bool(combined_readiness.iloc[0]["ready_from_warmup_start"]),
            "bootstrap_formal_ready": False,
            "diagnostic_bootstrap_only": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "outputs": {
                "row_breakdown": "pool1_warmup_blocker_breakdown.csv",
                "summary": "pool1_warmup_blocker_summary.csv",
                "options": "warmup_policy_options.csv",
                "options_md": "warmup_policy_options.md",
                "diagnostic_table": "diagnostic_bootstrap_candidate_warmup_table.csv",
                "combined_readiness": "combined_formal_target_stream_warmup_readiness.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "next_step_handoff.md",
                "final_summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_text(manifest, blocker_summary, option_table), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_diagnostic_recommendation", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _load_pool1_prices(
    price_cache_dir: str | Path,
    registry: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[str]]:
    prices: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for ticker in POOL1_TICKERS:
        frame, source_meta = _load_price_source(ticker, price_cache_dir=price_cache_dir, registry=registry)
        if frame is None:
            missing.append(ticker)
            continue
        prices[ticker] = frame
        meta[ticker] = source_meta
    return prices, meta, missing


def _blocked_row_breakdown(
    blocked: pd.DataFrame,
    lifecycle: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    lifecycle_by_ticker = {str(row["ticker"]): row for row in lifecycle.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for item in blocked.to_dict(orient="records"):
        signal_date = pd.Timestamp(str(item.get("signal_date"))).normalize()
        ticker_counts = _ticker_history_counts(signal_date, prices, lifecycle_by_ticker)
        ready = [ticker for ticker, count in ticker_counts.items() if count >= MIN_POOL1_HISTORY_ROWS]
        available_count = int(item.get("available_universe_count") or 0)
        missing_count = max(0, MIN_DYNAMIC_POOL1_UNIVERSE - available_count)
        rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "available_universe_count": available_count,
                "required_dynamic_universe_count": MIN_DYNAMIC_POOL1_UNIVERSE,
                "missing_candidate_count_to_minimum": missing_count,
                "ready_tickers_by_history": "|".join(ready),
                "ready_ticker_count_by_history": len(ready),
                "max_history_rows": max(ticker_counts.values()) if ticker_counts else 0,
                "min_required_history_rows": MIN_POOL1_HISTORY_ROWS,
                "missing_20d": False,
                "missing_60d": len(ready) < MIN_DYNAMIC_POOL1_UNIVERSE,
                "missing_persistence": True,
                "missing_attack_gate": True,
                "missing_benchmark": False,
                "missing_00631l_history": ticker_counts.get("00631L.TW", 0) < MIN_POOL1_HISTORY_ROWS,
                "candidate_universe_insufficient": available_count < MIN_DYNAMIC_POOL1_UNIVERSE,
                "blocker_category": _blocker_category(available_count),
                "blocker_reason_zh": _blocker_reason_zh(available_count),
                "formal_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _ticker_history_counts(
    signal_date: pd.Timestamp,
    prices: dict[str, pd.DataFrame],
    lifecycle_by_ticker: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ticker in POOL1_TICKERS:
        frame = prices.get(ticker)
        if frame is None:
            counts[ticker] = 0
            continue
        lifecycle = lifecycle_by_ticker.get(ticker, {})
        first_tradable = pd.Timestamp(str(lifecycle.get("first_tradable_date") or "1900-01-01")).normalize()
        if signal_date < first_tradable:
            counts[ticker] = 0
            continue
        history = frame.loc[frame.index <= signal_date, "adj_close"].dropna()
        counts[ticker] = int(len(history))
    return counts


def _blocker_category(available_count: int) -> str:
    if available_count <= 0:
        return "no_pool1_candidate_has_60d_warmup"
    if available_count < MIN_DYNAMIC_POOL1_UNIVERSE:
        return "partial_universe_below_minimum_for_state_machine"
    return "unknown_pool1_warmup_blocker"


def _blocker_reason_zh(available_count: int) -> str:
    if available_count <= 0:
        return "所有 Pool1 標的都還沒累積滿正式 60 日相對強度所需歷史，不能產正式攻擊候選。"
    if available_count < MIN_DYNAMIC_POOL1_UNIVERSE:
        return "只有少數標的滿足 60 日 warmup，候選池不足以啟動現行 Pool1 狀態機。"
    return "需人工檢查。"


def _blocker_summary(row_breakdown: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        row_breakdown.groupby("blocker_category", as_index=False)
        .agg(
            rows=("signal_date", "count"),
            first_signal_date=("signal_date", "min"),
            last_signal_date=("signal_date", "max"),
            max_available_universe_count=("available_universe_count", "max"),
        )
        .sort_values("first_signal_date")
    )
    grouped["formal_ready"] = False
    return grouped


def _candidate_warmup_table(row_breakdown: pd.DataFrame) -> pd.DataFrame:
    frame = row_breakdown.copy()
    return frame[
        [
            "signal_date",
            "available_universe_count",
            "ready_tickers_by_history",
            "max_history_rows",
            "min_required_history_rows",
            "blocker_category",
            "blocker_reason_zh",
            "formal_ready",
        ]
    ]


def _policy_options(row_breakdown: pd.DataFrame) -> pd.DataFrame:
    blocked_period = _period(row_breakdown, "signal_date")
    return pd.DataFrame(
        [
            {
                "option_id": "A_start_formal_replay_2015_01_28",
                "label_zh": "正式長區間從 2015-01-28 起跑，前 60 rows 標 warmup-only / non-tradable",
                "recommended": True,
                "formal_ready": True,
                "affected_period": blocked_period,
                "rigor_impact": "保留現行 60 日 lookback 與動態 universe 門檻，嚴謹性最高。",
                "future_data_risk": "無。只用當日前已累積資料；不補未滿 lookback 的分數。",
                "implementation_note": "把 2014-11-03～2015-01-27 作為資料暖機期，不納入正式可交易績效；combined target stream 可從 2015-01-28 接續。",
            },
            {
                "option_id": "B_bootstrap_first_60_rows_diagnostic_only",
                "label_zh": "補 bootstrap contract，嘗試讓 2014-11-03～2015-01-27 也產候選",
                "recommended": False,
                "formal_ready": False,
                "affected_period": blocked_period,
                "rigor_impact": "會降低嚴謹性，因為必須縮短 lookback、使用部分歷史或人工 seed state。",
                "future_data_risk": "若用後續資料補前段狀態會有 future-data 風險；若只用短歷史則與現行 formal contract 不等價。",
                "implementation_note": "只能產 report-only / diagnostic bootstrap table，不得包裝成 formal-ready，也不得拿去宣稱正式長區間績效。",
            },
        ]
    )


def _combined_readiness(
    row_breakdown: pd.DataFrame,
    pool1_replay: pd.DataFrame,
    pool2_daily: pd.DataFrame,
    pool2_eligible: pd.DataFrame,
) -> pd.DataFrame:
    first_replay = str(pool1_replay["signal_date"].min()) if not pool1_replay.empty and "signal_date" in pool1_replay.columns else ""
    last_replay = str(pool1_replay["signal_date"].max()) if not pool1_replay.empty and "signal_date" in pool1_replay.columns else ""
    pool2_first = str(pool2_daily["signal_date"].min()) if not pool2_daily.empty and "signal_date" in pool2_daily.columns else ""
    pool2_last = str(pool2_daily["signal_date"].max()) if not pool2_daily.empty and "signal_date" in pool2_daily.columns else ""
    return pd.DataFrame(
        [
            {
                "full_2014_11_combined_stream_ready": False,
                "ready_from_warmup_start": bool(first_replay == "2015-01-28" and not pool2_daily.empty),
                "recommended_formal_start_date": "2015-01-28",
                "warmup_only_period": _period(row_breakdown, "signal_date"),
                "first_pool1_replay_date": first_replay,
                "last_pool1_replay_date": last_replay,
                "pool2_daily_first_date": pool2_first,
                "pool2_daily_last_date": pool2_last,
                "pool2_eligible_rows": int(len(pool2_eligible)),
                "decision": "Full 2014-11 start is not formal-ready. A 2015-01-28 formal start is technically consistent if user/Research accepts warmup-only exclusion.",
                "formal_model_changed": False,
                "trade_decision_changed": False,
            }
        ]
    )


def _source_decision(price_meta: dict[str, dict[str, Any]], missing_prices: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "pool1_price_history",
                "status": "accepted" if not missing_prices else "partial",
                "decision": "Pool1 price sources are sufficient to identify warmup blocker; no proxy is promoted to formal.",
                "formal_or_proxy": "formal_input_for_warmup_diagnostic",
                "metadata": json.dumps({"missing_prices": missing_prices, "price_meta": price_meta}, ensure_ascii=False),
            },
            {
                "source_layer": "bootstrap_candidate_table",
                "status": "diagnostic_only",
                "decision": "Candidate warmup table is for diagnosis only and cannot be used as formal target stream.",
                "formal_or_proxy": "report_only_diagnostic",
                "metadata": "",
            },
        ]
    )


def _options_markdown(option_table: pd.DataFrame) -> str:
    lines = ["# Pool1 warmup/bootstrap policy options", ""]
    for item in option_table.to_dict(orient="records"):
        lines.extend(
            [
                f"## {item['option_id']}",
                f"- {item['label_zh']}",
                f"- recommended: {str(item['recommended']).lower()}",
                f"- formal_ready: {str(item['formal_ready']).lower()}",
                f"- affected_period: {item['affected_period']}",
                f"- rigor_impact: {item['rigor_impact']}",
                f"- future_data_risk: {item['future_data_risk']}",
                f"- implementation_note: {item['implementation_note']}",
                "",
            ]
        )
    return "\n".join(lines)


def _handoff_text(combined_readiness: pd.DataFrame) -> str:
    row = combined_readiness.iloc[0].to_dict()
    return "\n".join(
        [
            "# Pool1 warmup/bootstrap diagnostic handoff",
            "",
            "## Recommendation",
            "Use Option A: start formal long-range replay at 2015-01-28 and mark 2014-11-03..2015-01-27 as warmup-only / non-tradable.",
            "",
            "## Why",
            "The blocked rows are caused by the existing 60-day Pool1 lookback and minimum dynamic universe requirement. Bootstrapping the first 60 rows would require changing the formal replay contract or using diagnostic-only partial history.",
            "",
            "## Next step",
            f"Build combined formal target stream from {row.get('recommended_formal_start_date')} using Pool1 replay + Pool2 batched reconstruction; keep warmup rows excluded from formal performance.",
            "",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
        ]
    ) + "\n"


def _summary_text(manifest: dict[str, Any], blocker_summary: pd.DataFrame, option_table: pd.DataFrame) -> str:
    summary_lines = "\n".join(
        f"- {row.blocker_category}: {row.rows} rows ({row.first_signal_date}..{row.last_signal_date})"
        for row in blocker_summary.itertuples()
    )
    recommended = option_table[option_table["recommended"].astype(str).str.lower().eq("true")].iloc[0]
    return f"""# Pool1 warmup/bootstrap policy diagnostic

- status: {manifest['status']}
- blocked rows: {manifest['blocked_rows']}
- blocked period: {manifest['blocked_period']}
- zero scoring universe rows: {manifest['rows_with_zero_scoring_universe']}
- partial scoring universe rows: {manifest['rows_with_partial_scoring_universe']}
- recommended policy: {manifest['recommended_policy']}
- bootstrap formal ready: false
- formal_model_changed=false
- trade_decision_changed=false

## Blocker breakdown

{summary_lines}

## Recommendation

{recommended['label_zh']}

理由：前 60 rows 是正式 60 日 lookback 的 warmup，不是價格源缺漏。若強行 bootstrap，必須改 lookback 或 seed state，會降低嚴謹性，也不能與現行 formal contract 等價。
"""


def _period(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return f"{frame[column].min()}..{frame[column].max()}"


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose Pool1 warmup/bootstrap policy for first 60 long-range rows.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--long-range-dir", default=DEFAULT_LONG_RANGE_DIR)
    parser.add_argument("--lifecycle-dir", default=DEFAULT_LIFECYCLE_DIR)
    parser.add_argument("--pool2-dir", default=DEFAULT_POOL2_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    args = parser.parse_args(argv)
    output = run_pool1_warmup_bootstrap_policy_diagnostic(
        output_dir=args.output_dir,
        long_range_dir=args.long_range_dir,
        lifecycle_dir=args.lifecycle_dir,
        pool2_dir=args.pool2_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
