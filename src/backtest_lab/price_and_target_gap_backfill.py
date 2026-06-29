from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.tw50_backfill_audit import (
    AUDIT_END,
    AUDIT_START,
    BENCHMARK_TICKERS,
    DEFAULT_CONSTITUENTS_PATH,
    DEFAULT_PRICE_ROOTS,
    _load_constituents,
    _price_coverage_matrix,
    _scan_price_sources,
)
from backtest_lab.tw50_pit_backfill import (
    ETF_00631L_201411_TRADING_EVIDENCE,
    ETF_00631L_201411_TRADING_STATUS,
    RADAR_SOURCE_ACQUISITION_REFERENCE,
)


TASK_ID = "TASK-BACKTEST-CORE-DATA-BACKFILL-PRICE-AND-TARGET-GAPS-PHASE3-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_data_backfill_price_and_target_gaps_phase3_20260629"
PRICE_GAP_START = "2014-11-01"
PRICE_GAP_END = "2015-12-31"
ETF_00631L = "00631L.TW"


def run_price_and_target_gap_backfill(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    price_roots: tuple[str | Path, ...] = DEFAULT_PRICE_ROOTS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "twse_stock_day_raw").mkdir(exist_ok=True)
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
        log("load_current_constituents", "started", str(constituents_path))
        constituents = _load_constituents(Path(constituents_path))
        tickers = sorted(set(BENCHMARK_TICKERS) | set(constituents["ticker"].dropna().astype(str)))

        log("scan_price_cache", "started", ",".join(str(root) for root in price_roots))
        price_sources = _scan_price_sources(tuple(Path(root) for root in price_roots))
        price_coverage = _price_coverage_matrix(tickers, price_sources)

        log("build_phase3_ledgers", "started", "")
        jobs_00631l = _build_00631l_price_jobs(price_coverage)
        provisional_jobs = _build_provisional_price_jobs(price_coverage, constituents)
        priority = _price_source_priority()
        target_gaps = _target_stream_reconstruction_gaps()
        readiness = _readiness_ledger(jobs_00631l, provisional_jobs, target_gaps)
        manifest = _manifest(constituents, jobs_00631l, provisional_jobs, target_gaps, output)

        log("write_outputs", "started", "")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        jobs_00631l.to_csv(output / "00631l_price_backfill_jobs.csv", index=False, encoding="utf-8-sig")
        provisional_jobs.to_csv(output / "provisional_universe_price_backfill_jobs.csv", index=False, encoding="utf-8-sig")
        priority.to_csv(output / "price_source_priority.csv", index=False, encoding="utf-8-sig")
        target_gaps.to_csv(output / "target_stream_reconstruction_gaps.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "readiness_ledger.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, readiness), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_price_and_target_gap_backfill", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _build_00631l_price_jobs(price_coverage: pd.DataFrame) -> pd.DataFrame:
    row = _coverage_row(price_coverage, ETF_00631L)
    months = pd.period_range(PRICE_GAP_START, PRICE_GAP_END, freq="M")
    rows: list[dict[str, Any]] = []
    for month in months:
        month_start = month.to_timestamp().strftime("%Y-%m-%d")
        month_end = month.to_timestamp(how="end").strftime("%Y-%m-%d")
        roc_year = month.year - 1911
        twse_date = f"{roc_year:03d}{month.month:02d}01"
        rows.append(
            {
                "ticker": ETF_00631L,
                "month": str(month),
                "month_start": month_start,
                "month_end": month_end,
                "source": "TWSE_STOCK_DAY",
                "twse_query_date": twse_date,
                "twse_stock_no": "00631L",
                "job_status": "ready_for_twse_monthly_fetch",
                "checkpoint_key": f"twse_stock_day_raw/00631L_{month}.csv",
                "local_cache_first_date": row.get("first_date", ""),
                "local_cache_last_date": row.get("last_date", ""),
                "local_cache_source": row.get("source", ""),
                "trading_status": ETF_00631L_201411_TRADING_STATUS,
                "trading_evidence": ETF_00631L_201411_TRADING_EVIDENCE,
                "synthetic_allowed": False,
                "formal_ready_after_fetch_if": "TWSE rows exist, adjusted close policy is documented, and normalized cache passes coverage check.",
            }
        )
    return pd.DataFrame(rows)


def _coverage_row(price_coverage: pd.DataFrame, ticker: str) -> dict[str, Any]:
    row = price_coverage[price_coverage["ticker"] == ticker]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _build_provisional_price_jobs(price_coverage: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    name_map = {str(row["ticker"]): str(row.get("name", "")) for row in constituents.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for row in price_coverage.to_dict(orient="records"):
        ticker = str(row["ticker"])
        formal_ready = bool(row.get("formal_ready", False))
        rows.append(
            {
                "ticker": ticker,
                "name": name_map.get(ticker, _benchmark_name(ticker)),
                "universe_scope": "provisional_current_tw50_plus_0050_00631l",
                "not_complete_historical_universe": True,
                "first_date": str(row.get("first_date", "")),
                "last_date": str(row.get("last_date", "")),
                "coverage_ratio_201411_202312": float(row.get("coverage_ratio", 0.0)),
                "adjusted_close_ready": bool(row.get("has_adjusted_close", False)),
                "local_status": str(row.get("status", "missing")),
                "job_status": _provisional_price_job_status(ticker, row),
                "source_priority": _source_priority_for_ticker(ticker),
                "gap_reason": _provisional_gap_reason(ticker, row),
                "local_source": str(row.get("source", "")),
                "checkpoint_key": f"price_backfill/{ticker.replace('.', '_')}_coverage.csv",
            }
        )
    return pd.DataFrame(rows).sort_values(["job_status", "ticker"]).reset_index(drop=True)


def _benchmark_name(ticker: str) -> str:
    return {"0050.TW": "元大台灣50", "00631L.TW": "元大台灣50正2"}.get(ticker, "")


def _source_priority_for_ticker(ticker: str) -> str:
    if ticker == ETF_00631L:
        return "TWSE_STOCK_DAY_real_price_first; existing_cache_merge; yfinance_cross_check"
    if ticker == "0050.TW":
        return "existing_adjusted_cache_first; yfinance_cross_check; TWSE_STOCK_DAY_raw_reference"
    return "existing_adjusted_cache_first; yfinance_or_twse_backfill; PIT_membership_later_required"


def _provisional_price_job_status(ticker: str, row: dict[str, Any]) -> str:
    if ticker == ETF_00631L and str(row.get("first_date", "")) > AUDIT_START:
        return "ready_for_00631l_real_price_backfill"
    if bool(row.get("formal_ready", False)):
        return "price_ready_in_local_cache"
    if str(row.get("status", "")) == "missing":
        return "missing_price_cache"
    return "needs_price_extension_or_adjusted_close_review"


def _provisional_gap_reason(ticker: str, row: dict[str, Any]) -> str:
    if ticker == ETF_00631L and str(row.get("first_date", "")) > AUDIT_START:
        return "00631L is confirmed traded in 2014-11; backfill true TWSE price rows for 2014-11 to 2015-12."
    if bool(row.get("formal_ready", False)):
        return ""
    if str(row.get("status", "")) == "missing":
        return "No local price cache found for provisional universe ticker."
    return f"Local coverage {row.get('first_date', '')} to {row.get('last_date', '')}; requires {AUDIT_START} to {AUDIT_END} adjusted close coverage."


def _price_source_priority() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "00631L_201411_201512",
                "priority_order": 1,
                "source": "TWSE_STOCK_DAY",
                "source_role": "real_price_primary_source",
                "checkpoint": "twse_stock_day_raw/00631L_YYYY-MM.csv",
                "notes": "RADAR confirmed 2014-11 official trading rows. Use true TWSE rows; never synthetic 0050x2.",
            },
            {
                "scope": "00631L_201411_201512",
                "priority_order": 2,
                "source": "existing_cache_merge",
                "source_role": "merge_with_2016_plus_local_cache",
                "checkpoint": "price_backfill/00631L_TW_merged.csv",
                "notes": "Merge only after duplicate/date/adjusted-close policy is checked.",
            },
            {
                "scope": "0050_and_current_tw50_provisional",
                "priority_order": 1,
                "source": "existing_adjusted_cache",
                "source_role": "local_cache_first",
                "checkpoint": "provisional_universe_price_backfill_jobs.csv",
                "notes": "Provisional current universe only. Final historical universe waits for PIT.",
            },
            {
                "scope": "current_tw50_missing_or_partial",
                "priority_order": 2,
                "source": "yfinance_or_twse_batch_backfill",
                "source_role": "price_gap_backfill_candidate",
                "checkpoint": "price_backfill/ticker_progress.csv",
                "notes": "Use checkpointed batches; do not start large unbounded download.",
            },
        ]
    )


def _target_stream_reconstruction_gaps() -> pd.DataFrame:
    rows = [
        {
            "layer": "pit_candidate_universe_by_date",
            "status": "blocked_missing_exact_or_source_backed_manual_pit",
            "required_input": "effective_date, active ticker list, source evidence",
            "why_price_only_is_not_enough": "Formal selector must know which tickers were eligible on each historical date.",
            "owner": "Radar/Data source acquisition then Core normalization",
            "next_action": "Acquire exact/manual PIT archive and normalize before target replay.",
        },
        {
            "layer": "adjusted_price_features",
            "status": "partial_provisional_jobs_ready",
            "required_input": "adjusted close, dividend/split policy, daily calendar coverage",
            "why_price_only_is_not_enough": "Prices are necessary for scores but do not define candidate eligibility or Pool2 confirmation.",
            "owner": "Core",
            "next_action": "Backfill 00631L true prices and provisional current universe gaps; final universe waits for PIT.",
        },
        {
            "layer": "pool1_candidate_ranking_scores",
            "status": "blocked_missing_2014_2021_replay_contract",
            "required_input": "candidate_rank, candidate_score, score margin, target formed reason",
            "why_price_only_is_not_enough": "Previous-best target stream needs reproducible ranking evidence, not only price availability.",
            "owner": "Core",
            "next_action": "Build formal candidate ranking / score margin contract after data readiness.",
        },
        {
            "layer": "pool2_confirmation_state",
            "status": "blocked_missing_2014_2021_replay_contract",
            "required_input": "Pool2 confirmation/risk state, disagreement reason, confirmation age",
            "why_price_only_is_not_enough": "Current formal route depends on Pool2 confirmation/risk control.",
            "owner": "Core",
            "next_action": "Replay Pool2 confirmation inputs with PIT-safe historical data.",
        },
        {
            "layer": "formal_target_stream",
            "status": "blocked_until_pit_price_signal_layers_ready",
            "required_input": "formal target, previous target, target change flag, route/source metadata",
            "why_price_only_is_not_enough": "Formal performance cannot exist without target stream evidence.",
            "owner": "Core",
            "next_action": "Rebuild target stream only after PIT, price, Pool1 ranking, and Pool2 confirmation are ready.",
        },
        {
            "layer": "execution_ledger",
            "status": "blocked_until_formal_target_stream_ready",
            "required_input": "same-day and next-day fills, costs, cash, holding transitions",
            "why_price_only_is_not_enough": "Execution replay needs formal targets first.",
            "owner": "Core",
            "next_action": "Run execution ledger after formal target stream is rebuilt.",
        },
    ]
    return pd.DataFrame(rows)


def _readiness_ledger(jobs_00631l: pd.DataFrame, provisional_jobs: pd.DataFrame, target_gaps: pd.DataFrame) -> pd.DataFrame:
    price_ready = int((provisional_jobs["job_status"] == "price_ready_in_local_cache").sum()) if not provisional_jobs.empty else 0
    return pd.DataFrame(
        [
            {
                "area": "00631l_real_price_backfill",
                "status": "jobs_ready_not_downloaded",
                "can_advance_without_exact_pit": True,
                "blocks_full_2014_2023_replay": True,
                "detail": f"{len(jobs_00631l)} monthly TWSE STOCK_DAY jobs prepared for {PRICE_GAP_START} to {PRICE_GAP_END}.",
            },
            {
                "area": "provisional_universe_price_jobs",
                "status": "jobs_ready_provisional_only",
                "can_advance_without_exact_pit": True,
                "blocks_full_2014_2023_replay": True,
                "detail": f"{price_ready}/{len(provisional_jobs)} provisional tickers already have full local adjusted coverage.",
            },
            {
                "area": "tw50_exact_pit",
                "status": "blocked_missing_exact_archive",
                "can_advance_without_exact_pit": False,
                "blocks_full_2014_2023_replay": True,
                "detail": "Still cannot label 2014-2023 historical universe exact until PIT archive is acquired.",
            },
            {
                "area": "target_stream_reconstruction",
                "status": "blocked_after_data_layers",
                "can_advance_without_exact_pit": False,
                "blocks_full_2014_2023_replay": True,
                "detail": f"{len(target_gaps)} target/signal layers documented; price/PIT alone is not sufficient.",
            },
        ]
    )


def _manifest(
    constituents: pd.DataFrame,
    jobs_00631l: pd.DataFrame,
    provisional_jobs: pd.DataFrame,
    target_gaps: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "phase3_price_and_target_gap_jobs_ready",
        "audit_start": AUDIT_START,
        "audit_end": AUDIT_END,
        "00631l_gap_start": PRICE_GAP_START,
        "00631l_gap_end": PRICE_GAP_END,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "large_download_started": False,
        "network_fetch_started": False,
        "can_resume": True,
        "output_dir": str(output.resolve()),
        "current_constituent_unique_tickers": int(constituents["ticker"].nunique()) if not constituents.empty else 0,
        "provisional_universe_price_job_count": int(len(provisional_jobs)),
        "00631l_monthly_twse_jobs": int(len(jobs_00631l)),
        "00631l_trading_status": ETF_00631L_201411_TRADING_STATUS,
        "00631l_trading_evidence": ETF_00631L_201411_TRADING_EVIDENCE,
        "radar_source_acquisition_reference": RADAR_SOURCE_ACQUISITION_REFERENCE,
        "pit_exact_archive_ready": False,
        "provisional_universe_is_complete_history": False,
        "synthetic_00631l_used": False,
        "target_stream_reconstruction_gap_count": int(len(target_gaps)),
        "fully_validated_2014_2023_replay_ready": False,
    }


def _summary_zh(manifest: dict[str, Any], readiness: pd.DataFrame) -> str:
    lines = [
        "# Price / Target Gap Backfill Phase 3",
        "",
        "## 結論",
        "",
        "- exact PIT 仍 blocked，但 Core 不能因此停住；本次已把可先做的價格與 target stream 缺口拆成可執行 job/ledger。",
        "- 00631L 已由 RADAR/Data 確認 2014/11 有官方 TWSE 日成交資料；目前缺口是 2014/11-2015 真實價格/source/cache 回補，不是未上市或不可交易。",
        "- provisional universe 只包含 0050、00631L 與目前已知 current TW50 50 檔，不代表完整歷史 universe。",
        "- 2014-2021 previous-best replay 不只缺價格，也缺 PIT candidate universe、Pool1 ranking score、Pool2 confirmation、formal target stream 與 execution ledger。",
        "",
        "## 可立即推進",
        "",
        "- 依 `00631l_price_backfill_jobs.csv` 用 TWSE STOCK_DAY 月度 checkpoint 回補 00631L 真實日資料。",
        "- 依 `provisional_universe_price_backfill_jobs.csv` 處理 current TW50 + 0050 + 00631L 的價格缺口；輸出仍需標 provisional。",
        "",
        "## 仍需外部 source / manual",
        "",
        "- 2014/11-2023/12 TW50 exact 或 source-backed manual PIT archive。",
        "",
        "## 需要 Research / Experiments 判讀",
        "",
        "- 2014-2021 formal target stream 重建後，才可判斷 previous-best / 新策略能否用完整第一區間做正式 replay。",
        "",
        "## Readiness",
        "",
    ]
    for row in readiness.to_dict(orient="records"):
        lines.append(f"- `{row['area']}`：{row['status']}；{row['detail']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build price and target gap backfill ledgers that can advance before exact PIT is ready.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--price-root", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    price_roots = tuple(args.price_root) if args.price_root else DEFAULT_PRICE_ROOTS
    output = run_price_and_target_gap_backfill(
        constituents_path=args.constituents_path,
        price_roots=tuple(Path(root) for root in price_roots),
        output_dir=args.output_dir,
    )
    print(f"PRICE_TARGET_GAP_BACKFILL_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
