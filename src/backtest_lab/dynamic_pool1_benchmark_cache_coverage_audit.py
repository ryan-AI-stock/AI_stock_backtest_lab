"""Audit local 0050/00631L benchmark cache coverage for Dynamic Pool1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-BENCHMARK-CACHE-COVERAGE-AUDIT-001"
DEFAULT_CONTEXT_DIR = Path("outputs/dynamic_pool1_explicit_benchmark_context_contract_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_benchmark_cache_coverage_audit_20260704")
PRIMARY_CACHE = {
    "0050": Path("backtest_cache/0050_TW.csv"),
    "00631L": Path("backtest_cache/00631L_TW.csv"),
}


def run_benchmark_cache_coverage_audit(
    *,
    repo_root: str | Path = ".",
    context_dir: str | Path = DEFAULT_CONTEXT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    context = _resolve(root, context_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness = pd.read_csv(context / "benchmark_readiness_summary.csv")
    readiness["candidate_month"] = readiness["candidate_month"].astype(str)
    source_map = _scan_benchmark_sources(root)
    coverage = _build_coverage_by_month(readiness, source_map)
    missing = _build_missing_months(coverage)
    repair = _build_join_repair_candidates(source_map, missing)
    future_audit = pd.DataFrame(
        [
            {
                "audit_item": "coverage_scan",
                "future_data_violation_count": 0,
                "status": "date ranges only; no forward return or strategy replay",
            },
            {
                "audit_item": "join_repair_candidates",
                "future_data_violation_count": 0,
                "status": "candidate sources require explicit validation before replacing primary cache",
            },
        ]
    )

    coverage.to_csv(output / "benchmark_cache_coverage_by_month.csv", index=False, encoding="utf-8-sig")
    source_map.to_csv(output / "benchmark_cache_source_map.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output / "benchmark_missing_months.csv", index=False, encoding="utf-8-sig")
    repair.to_csv(output / "join_repair_candidates.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_benchmark_cache_coverage_audit",
        "output_dir": str(output),
        "source_context_dir": str(context),
        "candidate_months": int(readiness["candidate_month"].nunique()),
        "source_rows": int(len(source_map)),
        "missing_month_rows": int(len(missing)),
        "join_repair_candidate_rows": int(len(repair)),
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "future_data_violation_count": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest, missing, repair), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_context_readiness", "status": "completed"},
            {"step": "scan_local_benchmark_sources", "status": "completed"},
            {"step": "build_coverage_and_repair_candidates", "status": "completed"},
            {"step": "write_outputs", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _scan_benchmark_sources(root: Path) -> pd.DataFrame:
    candidates: list[Path] = []
    for pattern in ["*0050*.csv", "*00631L*.csv", "*00631l*.csv"]:
        candidates.extend(root.rglob(pattern))
    registry = _load_registry(root)
    rows = []
    seen: set[Path] = set()
    for path in sorted(set(candidates)):
        if path in seen or any(part == ".git" for part in path.parts):
            continue
        seen.add(path)
        ticker = _infer_ticker(path)
        if ticker not in {"0050", "00631L"}:
            continue
        stats = _date_stats(path)
        if not stats:
            continue
        rel = path.relative_to(root).as_posix()
        registry_row = registry.get(rel, {})
        rows.append(
            {
                "ticker": ticker,
                "path": rel,
                "source_class": _source_class(rel, registry_row),
                "first_date": stats["first_date"],
                "last_date": stats["last_date"],
                "row_count": stats["row_count"],
                "month_count": stats["month_count"],
                "is_primary_context_cache": rel == PRIMARY_CACHE[ticker].as_posix(),
                "price_source_ready": registry_row.get("price_source_ready", ""),
                "strategy_ready": registry_row.get("strategy_ready", ""),
                "provenance": registry_row.get("provenance", ""),
                "notes": registry_row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _load_registry(root: Path) -> dict[str, dict]:
    path = root / "data/price_source_registry.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    registry = {}
    for row in df.to_dict(orient="records"):
        source_path = str(row.get("source_path", "")).replace("\\", "/")
        if source_path:
            registry[source_path] = row
    return registry


def _infer_ticker(path: Path) -> str:
    text = path.name.lower()
    if "00631l" in text:
        return "00631L"
    if "0050" in text:
        return "0050"
    return ""


def _date_stats(path: Path) -> dict | None:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    date_cols = [col for col in header.columns if str(col).lower() in {"date", "trade_date", "signal_date"}]
    if not date_cols:
        return None
    try:
        df = pd.read_csv(path, usecols=[date_cols[0]])
    except Exception:
        return None
    dates = pd.to_datetime(df[date_cols[0]], errors="coerce").dropna()
    if dates.empty:
        return None
    return {
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "row_count": int(len(dates)),
        "month_count": int(dates.dt.to_period("M").nunique()),
    }


def _source_class(rel: str, registry_row: dict) -> str:
    if registry_row:
        return "registered_price_source"
    if rel.startswith("data/normalized_prices/"):
        return "normalized_price_candidate"
    if "/ad_hoc_" in rel or rel.startswith("backtest_cache/ad_hoc_"):
        return "ad_hoc_cache_candidate"
    if rel.startswith("backtest_cache/"):
        return "backtest_cache_candidate"
    if rel.startswith("outputs/"):
        return "output_evidence_or_intermediate"
    return "unknown_csv_candidate"


def _months_between(first_date: str, last_date: str) -> set[str]:
    start = pd.to_datetime(first_date).to_period("M")
    end = pd.to_datetime(last_date).to_period("M")
    return set(pd.period_range(start, end, freq="M").astype(str))


def _build_coverage_by_month(readiness: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    primary_months = {ticker: set() for ticker in ["0050", "00631L"]}
    any_months = {ticker: set() for ticker in ["0050", "00631L"]}
    for row in source_map.to_dict(orient="records"):
        months = _months_between(row["first_date"], row["last_date"])
        any_months[row["ticker"]].update(months)
        if bool(row.get("is_primary_context_cache", False)):
            primary_months[row["ticker"]].update(months)
    out = readiness.copy()
    out["primary_0050_cache_month_available"] = out["candidate_month"].isin(primary_months["0050"])
    out["primary_00631l_cache_month_available"] = out["candidate_month"].isin(primary_months["00631L"])
    out["any_0050_local_source_month_available"] = out["candidate_month"].isin(any_months["0050"])
    out["any_00631l_local_source_month_available"] = out["candidate_month"].isin(any_months["00631L"])
    out["join_repair_possible"] = (
        ((out["explicit_0050_ready_rate"] == 0) & out["any_0050_local_source_month_available"])
        | ((out["explicit_00631l_ready_rate"] == 0) & out["any_00631l_local_source_month_available"])
    )
    out["coverage_gap_reason"] = out.apply(_coverage_gap_reason, axis=1)
    return out


def _coverage_gap_reason(row: pd.Series) -> str:
    reasons = []
    if float(row.get("explicit_0050_ready_rate", 0)) == 0:
        reasons.append(
            "0050_primary_context_cache_missing_month"
            if not bool(row.get("primary_0050_cache_month_available", False))
            else "0050_month_available_but_trailing_window_or_join_not_ready"
        )
    if float(row.get("explicit_00631l_ready_rate", 0)) == 0:
        reasons.append(
            "00631L_primary_context_cache_missing_month"
            if not bool(row.get("primary_00631l_cache_month_available", False))
            else "00631L_month_available_but_trailing_window_or_join_not_ready"
        )
    return ";".join(reasons)


def _build_missing_months(coverage: pd.DataFrame) -> pd.DataFrame:
    mask = (coverage["explicit_0050_ready_rate"] < 1) | (coverage["explicit_00631l_ready_rate"] < 1)
    return coverage.loc[mask].copy()


def _build_join_repair_candidates(source_map: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing_months = set(missing["candidate_month"].astype(str))
    for row in source_map.to_dict(orient="records"):
        months = _months_between(row["first_date"], row["last_date"])
        overlap = sorted(months & missing_months)
        if not overlap or row.get("is_primary_context_cache"):
            continue
        rows.append(
            {
                "ticker": row["ticker"],
                "path": row["path"],
                "source_class": row["source_class"],
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "missing_month_overlap_count": len(overlap),
                "first_repairable_month": overlap[0],
                "last_repairable_month": overlap[-1],
                "recommended_next_step": _repair_step(row),
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["ticker", "missing_month_overlap_count"], ascending=[True, False])


def _repair_step(row: dict) -> str:
    source_class = row.get("source_class", "")
    if source_class == "registered_price_source":
        return "join_repair_candidate_registered_source_validate_adjustment_policy"
    if source_class == "ad_hoc_cache_candidate":
        return "validate_cache_provenance_before_join_repair_or_request_radar_source_package"
    if source_class == "normalized_price_candidate":
        return "join_repair_candidate_price_only_boundary"
    return "manual_review_before_join_repair"


def _summary_text(manifest: dict, missing: pd.DataFrame, repair: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 benchmark cache coverage audit",
            "",
            "本包只稽核 0050 / 00631L benchmark cache 覆蓋與 join repair 可能性，不跑策略、不改正式模型、不改報告。",
            "",
            f"- candidate months：{manifest['candidate_months']}",
            f"- source rows：{manifest['source_rows']}",
            f"- missing/partial month rows：{manifest['missing_month_rows']}",
            f"- join repair candidate rows：{manifest['join_repair_candidate_rows']}",
            "- 2023-04 前缺口主因是 primary benchmark cache 不覆蓋 2015-2023 早段；部分本機 ad-hoc/registered source 可能可作 join repair，但需先驗證 provenance/adjustment policy。",
            "- 2026-05 後缺口主因是 primary 0050/00631L cache 尚未更新到最新月份；若無 accepted local source，應交 Radar/Data 補最小 benchmark cache 包。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--context-dir", default=str(DEFAULT_CONTEXT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_benchmark_cache_coverage_audit(
        repo_root=args.repo_root,
        context_dir=args.context_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
