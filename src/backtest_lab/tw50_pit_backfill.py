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


DEFAULT_OUTPUT_DIR = "outputs/core_data_backfill_tw50_201411_202312_phase2_runner_20260629"
TASK_ID = "TASK-BACKTEST-CORE-DATA-BACKFILL-TW50-201411-202312-PHASE2-RUNNER-20260629"
AUDIT_REFERENCE = "outputs/core_data_backfill_tw50_201411_202312_audit_20260629"


def run_tw50_pit_backfill(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    price_roots: tuple[str | Path, ...] = DEFAULT_PRICE_ROOTS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase: str = "plan",
) -> Path:
    if phase != "plan":
        raise ValueError("Only phase='plan' is implemented. PIT normalize/price backfill must wait for accepted raw sources.")
    output = Path(output_dir)
    _prepare_output_dirs(output)
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

        log("scan_existing_price_cache", "started", ",".join(str(root) for root in price_roots))
        price_sources = _scan_price_sources(tuple(Path(root) for root in price_roots))
        price_coverage = _price_coverage_matrix(tickers, price_sources)

        log("build_phase2_contracts", "started", "")
        pit_sources = _pit_source_acquisition_plan()
        price_jobs = _provisional_price_backfill_jobs(price_coverage, constituents)
        job_plan = _backfill_job_plan(price_jobs, constituents)
        readiness = _readiness_ledger(price_jobs, constituents)
        raw_template = _raw_source_archive_template()
        pit_template = _manual_pit_constituents_template()
        pit_sample = _normalized_pit_sample(constituents)
        data_contract = _data_directory_contract(output)
        manifest = _manifest(constituents, price_jobs, pit_sources, readiness, output)

        log("write_outputs", "started", "")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        job_plan.to_csv(output / "backfill_job_plan.csv", index=False, encoding="utf-8-sig")
        pit_sources.to_csv(output / "pit_source_acquisition_plan.csv", index=False, encoding="utf-8-sig")
        price_jobs.to_csv(output / "provisional_price_backfill_jobs.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "readiness_ledger.csv", index=False, encoding="utf-8-sig")
        price_jobs.to_csv(output / "updated_price_coverage_panel.csv", index=False, encoding="utf-8-sig")
        raw_template.to_csv(output / "raw_source_archive_manifest_template.csv", index=False, encoding="utf-8-sig")
        pit_template.to_csv(output / "manual_pit_constituents_template.csv", index=False, encoding="utf-8-sig")
        pit_sample.to_csv(output / "normalized_pit_constituents_sample.csv", index=False, encoding="utf-8-sig")
        (output / "data_directory_contract.json").write_text(json.dumps(data_contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, job_plan, readiness), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve()), "phase": phase}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_tw50_pit_backfill", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _prepare_output_dirs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for child in ("raw_source_archive", "normalized", "price_backfill", "checkpoints"):
        (output / child).mkdir(parents=True, exist_ok=True)


def _pit_source_acquisition_plan() -> pd.DataFrame:
    rows = [
        {
            "source_id": "ftse_russell_tw50_constituent_notices",
            "source_type": "exact_candidate",
            "coverage_needed": f"{AUDIT_START} to {AUDIT_END}",
            "acquisition_mode": "manual_or_data_thread_archive",
            "raw_archive_path": "raw_source_archive/ftse_russell/",
            "normalization_status": "blocked_until_raw_archive_exists",
            "formal_ready_if": "effective_date and constituent list can be evidenced from original notice/report",
            "owner": "Radar/Data support recommended",
            "notes": "Priority source. Do not infer missing historical dates from current constituents.",
        },
        {
            "source_id": "twse_or_taiwan_index_constituent_notice_archive",
            "source_type": "exact_candidate",
            "coverage_needed": f"{AUDIT_START} to {AUDIT_END}",
            "acquisition_mode": "manual_or_data_thread_archive",
            "raw_archive_path": "raw_source_archive/twse_or_index/",
            "normalization_status": "blocked_until_raw_archive_exists",
            "formal_ready_if": "source provides point-in-time effective constituent changes",
            "owner": "Radar/Data support recommended",
            "notes": "Use as exact cross-check or fallback if FTSE archive is incomplete.",
        },
        {
            "source_id": "yuanta_0050_monthly_or_annual_reports",
            "source_type": "manual_evidence_candidate",
            "coverage_needed": f"{AUDIT_START} to {AUDIT_END}",
            "acquisition_mode": "manual_raw_archive_then_normalize",
            "raw_archive_path": "raw_source_archive/yuanta_0050_reports/",
            "normalization_status": "blocked_until_raw_archive_exists",
            "formal_ready_if": "report date, constituent names/tickers, and effective period are auditable",
            "owner": "Core can normalize after source archive is supplied",
            "notes": "Can seed manual PIT ledger if official change notices are not complete.",
        },
        {
            "source_id": "repo_current_snapshot_data_tw50_constituents_csv",
            "source_type": "proxy_current_snapshot",
            "coverage_needed": "current snapshot only",
            "acquisition_mode": "already_available",
            "raw_archive_path": "data/tw50_constituents.csv",
            "normalization_status": "available_proxy_not_history",
            "formal_ready_if": "never for 2014 historical PIT by itself",
            "owner": "Core",
            "notes": "Allowed as provisional price universe only; not accepted as historical exact constituents.",
        },
    ]
    return pd.DataFrame(rows)


def _provisional_price_backfill_jobs(price_coverage: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    name_map = {str(row["ticker"]): str(row.get("name", "")) for row in constituents.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for row in price_coverage.to_dict(orient="records"):
        ticker = str(row["ticker"])
        first_date = str(row.get("first_date", ""))
        status = str(row.get("status", "missing"))
        missing_reason = _price_missing_reason(ticker, row)
        job_status = _price_job_status(ticker, row)
        rows.append(
            {
                "ticker": ticker,
                "name": name_map.get(ticker, _benchmark_name(ticker)),
                "universe_scope": "provisional_current_tw50_plus_benchmarks",
                "formal_universe_status": "provisional_not_complete_history",
                "first_date": first_date,
                "last_date": str(row.get("last_date", "")),
                "coverage_ratio": float(row.get("coverage_ratio", 0.0)),
                "adjusted_close_ready": bool(row.get("has_adjusted_close", False)),
                "current_cache_status": status,
                "job_status": job_status,
                "missing_reason": missing_reason,
                "source": str(row.get("source", "")),
                "checkpoint_key": f"price_backfill/{ticker.replace('.', '_')}.csv",
                "can_run_before_pit": ticker in BENCHMARK_TICKERS or ticker in name_map,
            }
        )
    return pd.DataFrame(rows).sort_values(["job_status", "ticker"]).reset_index(drop=True)


def _benchmark_name(ticker: str) -> str:
    return {"0050.TW": "元大台灣50", "00631L.TW": "元大台灣50正2"}.get(ticker, "")


def _price_job_status(ticker: str, row: dict[str, Any]) -> str:
    if ticker == "00631L.TW" and str(row.get("first_date", "")) > AUDIT_START:
        return "needs_inception_or_source_review"
    if bool(row.get("formal_ready", False)):
        return "price_ready_skip_download"
    if str(row.get("status", "")) == "missing":
        return "needs_price_backfill"
    return "needs_price_extension_or_adjusted_close_review"


def _price_missing_reason(ticker: str, row: dict[str, Any]) -> str:
    if ticker == "00631L.TW" and str(row.get("first_date", "")) > AUDIT_START:
        return "00631L local coverage starts after 2014-11; confirm listing/inception before treating 2014-2015 as missing price data."
    if bool(row.get("formal_ready", False)):
        return ""
    if str(row.get("status", "")) == "missing":
        return "No local adjusted price cache found in scanned roots."
    return f"Local cache coverage is {row.get('first_date', '')} to {row.get('last_date', '')}; requires {AUDIT_START} to {AUDIT_END} adjusted close coverage."


def _backfill_job_plan(price_jobs: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    partial_or_missing = int((price_jobs["job_status"] != "price_ready_skip_download").sum()) if not price_jobs.empty else 0
    current_count = int(constituents["ticker"].nunique()) if not constituents.empty else 0
    rows = [
        {
            "step_order": 1,
            "step": "initialize_backfill_workspace",
            "status": "completed",
            "checkpoint": "current_step.txt",
            "resume_command": "python -m backtest_lab.tw50_pit_backfill --phase plan",
            "detail": "Created raw_source_archive, normalized, price_backfill, and checkpoints folders under this output root.",
        },
        {
            "step_order": 2,
            "step": "archive_exact_or_manual_pit_sources",
            "status": "blocked_waiting_source_archive",
            "checkpoint": "raw_source_archive_manifest_template.csv",
            "resume_command": "Add raw source files, fill raw_source_archive_manifest_template.csv, then rerun phase plan or future normalize-pit phase.",
            "detail": "Need PIT source evidence before computing exact historical ticker universe.",
        },
        {
            "step_order": 3,
            "step": "normalize_pit_constituents",
            "status": "blocked_by_step_2",
            "checkpoint": "manual_pit_constituents_template.csv",
            "resume_command": "Future phase: python -m backtest_lab.tw50_pit_backfill --phase normalize-pit",
            "detail": "Normalize effective_date/ticker/name/source_type/evidence; current snapshot must remain proxy.",
        },
        {
            "step_order": 4,
            "step": "build_provisional_price_backfill_jobs",
            "status": "completed_plan_only",
            "checkpoint": "provisional_price_backfill_jobs.csv",
            "resume_command": "Use job_status to batch safe price downloads after source approval; do not call this complete historical universe.",
            "detail": f"Built provisional jobs for current known TW50 plus benchmarks; current constituent minimum={current_count}, non-ready price jobs={partial_or_missing}.",
        },
        {
            "step_order": 5,
            "step": "backfill_final_pit_universe_prices",
            "status": "blocked_until_pit_universe_exists",
            "checkpoint": "price_backfill/final_pit_universe_progress.csv",
            "resume_command": "Future phase: python -m backtest_lab.tw50_pit_backfill --phase price-backfill --batch-size N",
            "detail": "Final ticker count is unknown until PIT archive is acquired and normalized.",
        },
        {
            "step_order": 6,
            "step": "rebuild_2014_2021_formal_target_stream",
            "status": "blocked_after_data_layer",
            "checkpoint": "formal_target_stream_readiness.csv",
            "resume_command": "Separate Core task after PIT, prices, and signal features are ready.",
            "detail": "Prices/PIT alone are insufficient; formal candidate/ranking/target evidence must be replayable.",
        },
    ]
    return pd.DataFrame(rows)


def _readiness_ledger(price_jobs: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    ready_prices = int((price_jobs["job_status"] == "price_ready_skip_download").sum()) if not price_jobs.empty else 0
    rows = [
        {
            "layer": "raw_source_archive",
            "status": "workspace_ready_sources_missing",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Radar/Data then Core",
            "gap": "Need auditable raw PIT source files for historical TW50 constituents.",
            "next_action": "Acquire/archive FTSE/TWSE/issuer source documents before normalization.",
        },
        {
            "layer": "pit_constituents_exact",
            "status": "blocked_missing_exact_pit",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core after source archive",
            "gap": "Only current/proxy snapshot exists; exact historical total ticker count is unknown.",
            "next_action": "Normalize accepted raw archive into effective-date constituent table.",
        },
        {
            "layer": "manual_pit_template",
            "status": "template_ready_not_formal_data",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": False,
            "owner": "Core",
            "gap": "Template is not evidence by itself.",
            "next_action": "Use only with source-backed manual ledger rows.",
        },
        {
            "layer": "provisional_price_jobs",
            "status": "plan_ready_provisional_universe_only",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core",
            "gap": f"{ready_prices}/{len(price_jobs)} provisional tickers are fully ready in local cache; universe excludes unknown historical entrants.",
            "next_action": "Run safe batch price backfill only after source policy is approved; final universe waits for PIT.",
        },
        {
            "layer": "00631l_inception_review",
            "status": "needs_inception_or_source_review",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core/Data",
            "gap": "Local 00631L coverage starts after 2014-11.",
            "next_action": "Confirm listing/inception and official availability before classifying 2014-2015 as missing or not applicable.",
        },
        {
            "layer": "formal_signal_feature_replay",
            "status": "blocked_after_pit_and_prices",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core",
            "gap": "Formal selector inputs and score/ranking evidence must be reconstructed for 2014-2021.",
            "next_action": "After data readiness, build formal candidate/ranking/target replay contract.",
        },
        {
            "layer": "formal_target_stream_rebuild",
            "status": "blocked_missing_2014_2021_target_evidence",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core",
            "gap": "Existing formal target stream starts at 2022-01-03.",
            "next_action": "Do not report 2014-2021 formal performance until target stream is rebuilt.",
        },
        {
            "layer": "execution_ledger_replay",
            "status": "blocked_after_formal_target_stream",
            "formal_ready": False,
            "blocks_full_2014_2023_replay": True,
            "owner": "Core",
            "gap": "Execution replay needs completed formal target stream first.",
            "next_action": "Rebuild same-day/next-day execution only after target stream exists.",
        },
    ]
    return pd.DataFrame(rows)


def _raw_source_archive_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_id",
            "raw_file_path",
            "source_url_or_reference",
            "document_date",
            "covered_effective_start",
            "covered_effective_end",
            "source_type",
            "checksum_sha256",
            "archived_at",
            "notes",
        ]
    )


def _manual_pit_constituents_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "effective_date",
            "ticker",
            "name",
            "source_type",
            "evidence_source_id",
            "evidence_file",
            "formal_ready",
            "notes",
        ]
    )


def _normalized_pit_sample(constituents: pd.DataFrame) -> pd.DataFrame:
    if constituents.empty:
        return _manual_pit_constituents_template()
    sample = constituents.head(10).copy()
    return pd.DataFrame(
        {
            "effective_date": sample.get("effective_date", ""),
            "ticker": sample.get("ticker", ""),
            "name": sample.get("name", ""),
            "source_type": "proxy_current_snapshot",
            "evidence_source_id": "repo_current_snapshot_data_tw50_constituents_csv",
            "evidence_file": "data/tw50_constituents.csv",
            "formal_ready": False,
            "notes": "Sample only. Current snapshot cannot be used as historical exact PIT.",
        }
    )


def _data_directory_contract(output: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_root": str(output),
        "raw_source_archive": {
            "path": "raw_source_archive/",
            "rule": "Only immutable raw downloaded/manual source files and source manifest; no normalized edits.",
        },
        "normalized": {
            "path": "normalized/",
            "rule": "Source-backed PIT constituent tables after exact/proxy/manual classification.",
        },
        "price_backfill": {
            "path": "price_backfill/",
            "rule": "Per-ticker price backfill progress and normalized adjusted close coverage panels.",
        },
        "checkpoints": {
            "path": "checkpoints/",
            "rule": "Small resumability ledgers for batch steps; never hide failed tickers.",
        },
    }


def _manifest(
    constituents: pd.DataFrame,
    price_jobs: pd.DataFrame,
    pit_sources: pd.DataFrame,
    readiness: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "phase2_runner_plan_ready",
        "phase": "plan",
        "audit_reference": AUDIT_REFERENCE,
        "audit_start": AUDIT_START,
        "audit_end": AUDIT_END,
        "output_dir": str(output.resolve()),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "large_download_started": False,
        "can_resume": True,
        "resume_command": f"python -m backtest_lab.tw50_pit_backfill --phase plan --output-dir {output.as_posix()}",
        "current_step": "completed",
        "raw_source_archive_ready": True,
        "pit_source_plan_rows": int(len(pit_sources)),
        "current_constituent_unique_tickers": int(constituents["ticker"].nunique()) if not constituents.empty else 0,
        "provisional_price_job_count": int(len(price_jobs)),
        "provisional_price_ready_count": int((price_jobs["job_status"] == "price_ready_skip_download").sum()) if not price_jobs.empty else 0,
        "exact_historical_total_ticker_count": "blocked_until_PIT_archive_acquired",
        "needs_radar_or_data_thread_support": True,
        "fully_validated_2014_2023_replay_ready": False,
        "missing_layers_for_full_previous_best_replay": readiness[readiness["blocks_full_2014_2023_replay"] == True]["layer"].tolist(),
    }


def _summary_zh(manifest: dict[str, Any], job_plan: pd.DataFrame, readiness: pd.DataFrame) -> str:
    lines = [
        "# TW50 / 0050 / 00631L 2014-11 至 2023-12 Phase 2 Backfill Runner",
        "",
        "## 本次交付",
        "",
        "- 這一棒不是重複 Phase 1 audit；已建立可續跑的資料補齊 runner、job plan、PIT source acquisition plan、price backfill job ledger 與 readiness ledger。",
        "- 本次沒有啟動大型下載，也沒有修改正式模型、正式 target 或交易決策。",
        "- 目前 output root 內已建立 `raw_source_archive/`、`normalized/`、`price_backfill/`、`checkpoints/`，後續可把人工或官方原始來源放入 raw archive 後接著 normalize。",
        "",
        "## 目前仍缺什麼",
        "",
    ]
    for row in readiness.to_dict(orient="records"):
        if bool(row["blocks_full_2014_2023_replay"]):
            lines.append(f"- `{row['layer']}`：{row['status']}。缺口：{row['gap']} 下一步：{row['next_action']}")
    lines.extend(
        [
            "",
            "## 如何續跑",
            "",
            f"- 目前可重跑 plan：`{manifest['resume_command']}`。",
            "- PIT source 取得後：填寫 `raw_source_archive_manifest_template.csv`，再用後續 `normalize-pit` phase 接上；目前此 phase 尚未啟用，避免在沒有來源時假造 PIT。",
            "- 價格補齊：先依 `provisional_price_backfill_jobs.csv` 分批處理 current TW50 + 0050 + 00631L；但這只是 provisional universe，不代表完整歷史 universe。",
            "",
            "## 完整 2014/11-2023/12 previous-best replay 還差幾層",
            "",
            "1. PIT：2014/11-2023/12 TW50 exact 或可稽核 manual constituent ledger。",
            "2. Price：PIT universe 全部 ticker 的 adjusted close / dividend / split coverage。",
            "3. Signal features：正式 selector 所需 score/ranking/confirmation input 的 2014-2021 重建。",
            "4. Target stream：用正式 contract 產生 2014-2021 formal target stream，不能用 current snapshot 或 proxy 回填。",
            "5. Execution ledger：target stream 完成後才重建 same-day / next-day execution replay。",
            "",
            "## 是否需要 Radar/Data 支援",
            "",
            "- 需要。Core 已建立 runner 與資料治理結構，但 PIT source acquisition 應交 Radar/Data 或資料專線協助取得 FTSE/TWSE/issuer historical constituent notices。",
        ]
    )
    lines.append("")
    lines.append("## Phase 2 job plan")
    lines.append("")
    for row in job_plan.to_dict(orient="records"):
        lines.append(f"{row['step_order']}. {row['step']}：{row['status']}；checkpoint={row['checkpoint']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable TW50 PIT/price backfill runner scaffolding.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--price-root", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase", default="plan")
    args = parser.parse_args()
    price_roots = tuple(args.price_root) if args.price_root else DEFAULT_PRICE_ROOTS
    output = run_tw50_pit_backfill(
        constituents_path=args.constituents_path,
        price_roots=tuple(Path(root) for root in price_roots),
        output_dir=args.output_dir,
        phase=args.phase,
    )
    print(f"TW50_PIT_BACKFILL_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
