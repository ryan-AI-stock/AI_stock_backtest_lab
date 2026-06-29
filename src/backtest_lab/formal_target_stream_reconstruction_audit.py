from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import get_formal_model_contract
from backtest_lab.supplemental_price_sources import DEFAULT_PRICE_SOURCE_REGISTRY, load_price_source_registry
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


TASK_ID = "TASK-BACKTEST-CORE-FORMAL-TARGET-STREAM-RECONSTRUCTION-PHASE7-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_formal_target_stream_reconstruction_phase7_20260629"
CURRENT_FORMAL_STREAM_START = "2022-01-03"
LATEST_KNOWN_FORMAL_DATE = "2026-06-12"


def run_formal_target_stream_reconstruction_audit(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    price_roots: tuple[str | Path, ...] = DEFAULT_PRICE_ROOTS,
    registry_path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path = ".",
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
        log("load_formal_contract", "started", "formal_model_contract.py")
        formal_contract = get_formal_model_contract()

        log("scan_price_and_constituent_inputs", "started", "")
        constituents = _load_constituents(Path(constituents_path))
        tickers = sorted(set(BENCHMARK_TICKERS) | set(constituents["ticker"].dropna().astype(str)))
        base_sources = _scan_price_sources(tuple(Path(root) for root in price_roots))
        base_coverage = _price_coverage_matrix(tickers, base_sources)
        supplemental = _load_supplemental_usage(registry_path)

        log("build_dependency_matrix", "started", "")
        code_paths = _required_code_paths(Path(repo_root))
        matrix = _dependency_matrix(formal_contract, constituents, base_coverage, supplemental)
        reconstructable = matrix[matrix["can_reconstruct_now"].isin(["yes", "partial"])].copy()
        missing = matrix[matrix["can_reconstruct_now"].eq("no")].copy()
        job_plan = _reconstruction_job_plan()
        blockers = _remaining_blockers(matrix)
        manifest = _manifest(output, formal_contract, matrix, blockers)

        log("write_outputs", "started", "")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        matrix.to_csv(output / "formal_target_stream_dependency_matrix.csv", index=False, encoding="utf-8-sig")
        reconstructable.to_csv(output / "reconstructable_inputs.csv", index=False, encoding="utf-8-sig")
        missing.to_csv(output / "missing_inputs.csv", index=False, encoding="utf-8-sig")
        job_plan.to_csv(output / "reconstruction_job_plan.csv", index=False, encoding="utf-8-sig")
        code_paths.to_csv(output / "required_code_paths.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "remaining_blockers.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, matrix, blockers), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_formal_target_stream_reconstruction_audit", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _load_supplemental_usage(registry_path: str | Path) -> pd.DataFrame:
    registry = load_price_source_registry(registry_path)
    if registry.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "source_path",
                "source_type",
                "coverage_start",
                "coverage_end",
                "price_source_ready",
                "strategy_ready",
                "synthetic_used",
            ]
        )
    return registry.copy()


def _required_code_paths(repo_root: Path) -> pd.DataFrame:
    rows = [
        (
            "formal_model_contract",
            "src/backtest_lab/formal_model_contract.py",
            "正式模型 target/route 與邊界 contract。",
            "selector_contract",
        ),
        (
            "stock_pool_observation_decision_first",
            "src/backtest_lab/stock_pool_observation.py",
            "每日正式報告、正式候選排名輸出、前一交易日 target contract。",
            "report_and_candidate_contract",
        ),
        (
            "stock_pool_formal_daily_replay",
            "src/backtest_lab/stock_pool_formal_daily_replay.py",
            "2022+ 正式 daily replay / target stream 相關口徑。",
            "historical_replay",
        ),
        (
            "data_readiness_with_supplemental",
            "src/backtest_lab/data_readiness_with_supplemental.py",
            "採用 supplemental price source 的 coverage/readiness 層。",
            "data_readiness",
        ),
        (
            "supplemental_price_sources",
            "src/backtest_lab/supplemental_price_sources.py",
            "00631L 2014/11-2015 補充價格 source adapter。",
            "price_source_adapter",
        ),
        (
            "remove_cap_next_day_validation",
            "src/backtest_lab/remove_cap_next_day_validation.py",
            "remove-cap next-day apples-to-apples validation runner。",
            "execution_validation",
        ),
        (
            "formal_absorb_pool1_pool2",
            "src/backtest_lab/formal_absorb_pool1_pool2.py",
            "Pool1+Pool2 formal absorption package historical code path。",
            "formal_absorption_evidence",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "code_path_name": name,
                "path": path,
                "exists": (repo_root / path).exists(),
                "role": role,
                "purpose": purpose,
            }
            for name, path, purpose, role in rows
        ]
    )


def _dependency_matrix(
    formal_contract: dict[str, Any],
    constituents: pd.DataFrame,
    base_coverage: pd.DataFrame,
    supplemental: pd.DataFrame,
) -> pd.DataFrame:
    row_0050 = _row_for_ticker(base_coverage, "0050.TW")
    row_00631l = _row_for_ticker(base_coverage, "00631L.TW")
    supplemental_00631l = _supplemental_for_ticker(supplemental, "00631L.TW")
    constituents_effective_dates = sorted(set(constituents.get("effective_date", pd.Series(dtype=str)).astype(str)))
    pit_coverage = ";".join(date for date in constituents_effective_dates if date) or "none"
    pit_status = _pit_status(constituents)
    rows: list[dict[str, Any]] = [
        {
            "dependency_name": "tw50_pit_candidate_universe_by_date",
            "required_for": "Pool1 / Pool2 / selector / report",
            "current_coverage_start": pit_coverage,
            "current_coverage_end": pit_coverage,
            "2014_2021_availability": "missing_exact_pit_archive",
            "source_path": str(DEFAULT_CONSTITUENTS_PATH),
            "missing_source": "2014/11-2023/12 dated official TW50/0050 constituent archive",
            "can_reconstruct_now": "no",
            "blocker_type": "PIT",
            "owner": "Radar/Data",
            "notes": f"現有成分資料狀態為 {pit_status}；不可用 current snapshot 反推 2014-2021 exact universe。",
        },
        {
            "dependency_name": "0050_price_history",
            "required_for": "Pool1 benchmark / market regime / report benchmark",
            "current_coverage_start": row_0050.get("first_date", ""),
            "current_coverage_end": row_0050.get("last_date", ""),
            "2014_2021_availability": "available_or_partial_from_local_cache",
            "source_path": row_0050.get("source", ""),
            "missing_source": "" if _as_bool(row_0050.get("formal_ready")) else "adjusted close or full date coverage may need verification",
            "can_reconstruct_now": "yes" if _as_bool(row_0050.get("formal_ready")) else "partial",
            "blocker_type": "price",
            "owner": "Core",
            "notes": "價格層可用性不等於 target stream 可重建；仍需 PIT 與 signal features。",
        },
        {
            "dependency_name": "00631L_price_history",
            "required_for": "Pool1 market exposure candidate / benchmark / execution price",
            "current_coverage_start": _combined_first(row_00631l, supplemental_00631l),
            "current_coverage_end": _combined_last(row_00631l, supplemental_00631l),
            "2014_2021_availability": "price_source_ready_with_supplemental_2014_11_2015",
            "source_path": _join_nonempty([row_00631l.get("source", ""), supplemental_00631l.get("source_path", "")]),
            "missing_source": "adjusted-close/distribution policy final validation",
            "can_reconstruct_now": "partial",
            "blocker_type": "price_policy",
            "owner": "Core",
            "notes": "00631L 真實價格補充 source 已可讀，但只能標 price-source ready，不代表 strategy-ready。",
        },
        {
            "dependency_name": "pool1_candidate_ranking_scores",
            "required_for": "Pool1 / selector / report score margin",
            "current_coverage_start": CURRENT_FORMAL_STREAM_START,
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "missing",
            "source_path": "src/backtest_lab/stock_pool_observation.py",
            "missing_source": "2014-2021 Pool1 signal features, ranking score inputs, and PIT candidate universe",
            "can_reconstruct_now": "no",
            "blocker_type": "signal_feature",
            "owner": "Core + Research/Experiments",
            "notes": "目前每日報告可輸出正式候選排名，但 2014-2021 尚無可重放的 Pool1 ranking panel。",
        },
        {
            "dependency_name": "pool1_score_margin_contract",
            "required_for": "selector diagnostics / report score margin / low-confidence studies",
            "current_coverage_start": CURRENT_FORMAL_STREAM_START,
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "missing_formal_panel",
            "source_path": "src/backtest_lab/stock_pool_observation.py",
            "missing_source": "formal_candidate_ranking_panel for 2014-2021 with rank2/rank3 score margins",
            "can_reconstruct_now": "no",
            "blocker_type": "target_contract",
            "owner": "Core",
            "notes": "不能再用 proxy top3/score gap 充正式 contract；需先產 formal candidate ranking panel。",
        },
        {
            "dependency_name": "pool2_confirmation_state",
            "required_for": "Pool2 / selector confirmation / risk layer",
            "current_coverage_start": CURRENT_FORMAL_STREAM_START,
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "missing",
            "source_path": "formal route: pool1_primary_pool2_confirmation",
            "missing_source": "2014-2021 Pool2 confirmation inputs and PIT-safe readiness evidence",
            "can_reconstruct_now": "no",
            "blocker_type": "signal_feature",
            "owner": "Core + Radar/Data",
            "notes": f"正式 Pool2 policy={formal_contract.get('pool2_policy', '')}；2014-2021 尚缺可重放 input。",
        },
        {
            "dependency_name": "formal_selector_target_contract",
            "required_for": "selector / formal target stream",
            "current_coverage_start": formal_contract.get("formal_model_effective_date", ""),
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "contract_exists_inputs_missing",
            "source_path": "src/backtest_lab/formal_model_contract.py",
            "missing_source": "historical Pool1 ranking + Pool2 confirmation panels",
            "can_reconstruct_now": "partial",
            "blocker_type": "target_contract",
            "owner": "Core",
            "notes": "正式 contract 已存在，但不能在 inputs 缺失時產假 target stream。",
        },
        {
            "dependency_name": "previous_formal_target_contract",
            "required_for": "report switch signal / execution diagnostics",
            "current_coverage_start": CURRENT_FORMAL_STREAM_START,
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "missing_until_formal_target_stream_exists",
            "source_path": "src/backtest_lab/stock_pool_observation.py",
            "missing_source": "2014-2021 formal target stream daily ledger",
            "can_reconstruct_now": "no",
            "blocker_type": "target_stream",
            "owner": "Core",
            "notes": "前一交易日 target contract 依賴已生成的 daily formal target stream。",
        },
        {
            "dependency_name": "execution_price_ledger_inputs",
            "required_for": "execution / next-day ledger / replay",
            "current_coverage_start": CURRENT_FORMAL_STREAM_START,
            "current_coverage_end": LATEST_KNOWN_FORMAL_DATE,
            "2014_2021_availability": "missing_until_target_stream_exists",
            "source_path": "execution ledgers from current formal replay outputs",
            "missing_source": "2014-2021 formal target stream and next-day fill calendar",
            "can_reconstruct_now": "no",
            "blocker_type": "execution",
            "owner": "Core",
            "notes": "價格資料先補齊仍不能直接跑 execution；需要 daily target action stream。",
        },
        {
            "dependency_name": "formal_report_decision_first_fields",
            "required_for": "report",
            "current_coverage_start": "latest_report",
            "current_coverage_end": "latest_report",
            "2014_2021_availability": "not_a_historical_blocker_after_target_stream",
            "source_path": "src/backtest_lab/stock_pool_observation.py",
            "missing_source": "historical target stream if report replay is needed",
            "can_reconstruct_now": "partial",
            "blocker_type": "report",
            "owner": "Core",
            "notes": "報告 contract 已可用；歷史回測仍需 selector target stream 先存在。",
        },
    ]
    return pd.DataFrame(rows)


def _reconstruction_job_plan() -> pd.DataFrame:
    rows = [
        (
            1,
            "freeze_current_formal_contract",
            "completed",
            "Use formal_model_contract.py as the target route contract; do not change selector.",
            "Core",
        ),
        (
            2,
            "acquire_tw50_exact_pit_archive",
            "blocked",
            "Need dated official TW50/0050 constituent archive for 2014/11-2023/12.",
            "Radar/Data",
        ),
        (
            3,
            "finalize_00631l_adjusted_price_policy",
            "pending",
            "Confirm distribution/adjusted-close policy for 00631L supplemental source before total-return replay.",
            "Core",
        ),
        (
            4,
            "build_pool1_candidate_ranking_panel_2014_2021",
            "blocked",
            "Requires PIT universe and Pool1 signal feature inputs; output rank, score, and margin to rank2/rank3.",
            "Core",
        ),
        (
            5,
            "build_pool2_confirmation_panel_2014_2021",
            "blocked",
            "Requires Pool2 PIT-safe confirmation inputs matching current formal contract.",
            "Core + Radar/Data",
        ),
        (
            6,
            "generate_formal_target_stream_2014_2021",
            "blocked",
            "Use only formal candidate ranking and Pool2 confirmation panels; no proxy target stream.",
            "Core",
        ),
        (
            7,
            "build_next_day_execution_ledger_2014_2021",
            "blocked",
            "Requires formal target stream and executable price calendar.",
            "Core",
        ),
        (
            8,
            "handoff_to_experiments_for_two_period_validation",
            "blocked",
            "Run apples-to-apples validation only after PIT, signal, target, and execution layers are ready.",
            "Experiments",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "step_order": order,
                "job_name": job,
                "status": status,
                "minimum_action": action,
                "owner": owner,
            }
            for order, job, status, action, owner in rows
        ]
    )


def _remaining_blockers(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in matrix[matrix["can_reconstruct_now"].ne("yes")].to_dict(orient="records"):
        if row["blocker_type"] in {"report"}:
            continue
        rows.append(
            {
                "blocker": row["dependency_name"],
                "blocker_type": row["blocker_type"],
                "severity": "blocks_2014_2021_strategy_replay" if row["can_reconstruct_now"] == "no" else "partial_ready_caveat",
                "detail": row["missing_source"] or row["notes"],
                "owner": row["owner"],
                "blocks_strategy_ready": row["can_reconstruct_now"] == "no",
            }
        )
    return pd.DataFrame(rows)


def _manifest(output: Path, formal_contract: dict[str, Any], matrix: pd.DataFrame, blockers: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_dependency_matrix_ready",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "formal_model_target": formal_contract.get("formal_model_target", ""),
        "formal_model_route": formal_contract.get("formal_model_route", ""),
        "target_stream_generated": False,
        "fake_target_stream_generated": False,
        "current_formal_stream_start": CURRENT_FORMAL_STREAM_START,
        "latest_known_formal_date": LATEST_KNOWN_FORMAL_DATE,
        "audit_start": AUDIT_START,
        "audit_end": AUDIT_END,
        "dependency_count": int(len(matrix)),
        "missing_dependency_count": int(matrix["can_reconstruct_now"].eq("no").sum()),
        "partial_dependency_count": int(matrix["can_reconstruct_now"].eq("partial").sum()),
        "remaining_blocker_count": int(len(blockers)),
        "strategy_ready": False,
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, Any], matrix: pd.DataFrame, blockers: pd.DataFrame) -> str:
    missing = matrix[matrix["can_reconstruct_now"].eq("no")]
    partial = matrix[matrix["can_reconstruct_now"].eq("partial")]
    lines = [
        "# Formal Target Stream Reconstruction Phase 7",
        "",
        "## 結論",
        "",
        "- 本任務沒有改正式 selector、formal target 或 trade decision。",
        "- 本任務沒有產生 2014-2021 假 target stream；只建立可重跑的依賴矩陣與施工順序。",
        f"- 目前正式模型：`{manifest['formal_model_target']}` / `{manifest['formal_model_route']}`。",
        "- 00631L 2014/11-2015 價格缺口已可由 supplemental source 看見，但這只代表 price-source ready。",
        "- 2014-2021 不能完整回測的主因不是單一價格缺口，而是 PIT universe、Pool1 ranking signal、Pool2 confirmation signal、formal target stream、execution ledger 仍未完整。",
        "",
        "## 2014-2021 缺口拆解",
        "",
    ]
    for row in missing.to_dict(orient="records"):
        lines.append(f"- `{row['dependency_name']}`：缺 `{row['missing_source']}`；blocker_type={row['blocker_type']}。")
    lines += [
        "",
        "## 可部分重建但不能直接當策略績效",
        "",
    ]
    for row in partial.to_dict(orient="records"):
        lines.append(f"- `{row['dependency_name']}`：{row['notes']}")
    lines += [
        "",
        "## 剩餘 blocker",
        "",
    ]
    for row in blockers.to_dict(orient="records"):
        lines.append(f"- `{row['blocker']}`：{row['detail']}")
    return "\n".join(lines) + "\n"


def _row_for_ticker(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if frame.empty or "ticker" not in frame.columns:
        return {}
    row = frame[frame["ticker"].astype(str).eq(ticker)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _supplemental_for_ticker(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    return _row_for_ticker(frame, ticker)


def _combined_first(base_row: dict[str, Any], supplemental_row: dict[str, Any]) -> str:
    candidates = [str(base_row.get("first_date", "")), str(supplemental_row.get("coverage_start", ""))]
    dates = [pd.Timestamp(value) for value in candidates if value]
    if not dates:
        return ""
    return min(dates).strftime("%Y-%m-%d")


def _combined_last(base_row: dict[str, Any], supplemental_row: dict[str, Any]) -> str:
    candidates = [str(base_row.get("last_date", "")), str(supplemental_row.get("coverage_end", ""))]
    dates = [pd.Timestamp(value) for value in candidates if value]
    if not dates:
        return ""
    return max(dates).strftime("%Y-%m-%d")


def _join_nonempty(values: list[Any]) -> str:
    return ";".join(str(value) for value in values if str(value))


def _pit_status(constituents: pd.DataFrame) -> str:
    if constituents.empty:
        return "missing"
    sources = " ".join(constituents.get("source", pd.Series(dtype=str)).astype(str)).lower()
    if "exact" in sources or "ftse_pdf" in sources:
        return "exact_or_exact_candidate"
    if "proxy" in sources or "snapshot" in sources or "seed" in sources:
        return "proxy_current_snapshot"
    return "unknown_or_manual"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dependencies for reconstructing 2014-2021 formal target stream.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--price-root", action="append", default=[])
    parser.add_argument("--registry-path", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    price_roots = tuple(args.price_root) if args.price_root else DEFAULT_PRICE_ROOTS
    output = run_formal_target_stream_reconstruction_audit(
        constituents_path=args.constituents_path,
        price_roots=tuple(Path(root) for root in price_roots),
        registry_path=args.registry_path,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(f"FORMAL_TARGET_STREAM_RECONSTRUCTION_AUDIT_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
