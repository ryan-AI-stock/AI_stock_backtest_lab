from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-SECTOR-PROXY-READINESS-EXPANSION-001"
DEFAULT_SECTOR_PROXY_OUTPUT = "outputs/dynamic_pool1_twse_sector_proxy_diagnostic_panel_20260704"
DEFAULT_OUTPUT_DIR = "outputs/dynamic_pool1_sector_proxy_readiness_expansion_20260704"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_PRICE_SOURCE_REGISTRY = "data/price_source_registry.csv"
DEFAULT_CANDIDATE_PANEL_2022_LATEST = (
    "outputs/dynamic_pool1_candidate_ranking_panel_2022_latest/candidate_ranking_panel_2022_latest.csv"
)


def run_dynamic_pool1_sector_proxy_readiness_expansion(
    *,
    sector_proxy_output: str | Path = DEFAULT_SECTOR_PROXY_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    candidate_panel_2022_latest: str | Path = DEFAULT_CANDIDATE_PANEL_2022_LATEST,
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
        sector_root = Path(sector_proxy_output)
        log("load_sector_proxy_panel", "started", str(sector_root))
        sector_panel = _read_csv_required(sector_root / "dynamic_pool1_twse_sector_proxy_panel.csv")
        sector_readiness = _load_json(sector_root / "dynamic_pool1_sector_proxy_readiness.json")

        log("audit_price_coverage", "started", str(price_cache_dir))
        price_files = _discover_price_files(Path(price_cache_dir), Path(price_source_registry))
        price_coverage = _build_price_coverage(sector_panel, price_files)
        price_blockers = _build_price_blockers(price_coverage)

        log("build_candidate_sector_context_2022_latest", "started", str(candidate_panel_2022_latest))
        candidate_context, candidate_blockers = _build_candidate_sector_context_2022_latest(
            sector_panel=sector_panel,
            candidate_panel_path=Path(candidate_panel_2022_latest),
        )

        log("write_outputs", "started", str(output))
        join_blockers = _join_blockers(price_blockers, candidate_blockers)
        readiness = _readiness_for_experiments(
            sector_readiness=sector_readiness,
            sector_panel=sector_panel,
            price_coverage=price_coverage,
            candidate_context=candidate_context,
            candidate_blockers=candidate_blockers,
        )
        future_audit = _future_data_violation_audit(sector_readiness)
        manifest = _manifest(
            output=output,
            sector_proxy_output=sector_root,
            price_cache_dir=Path(price_cache_dir),
            candidate_panel_2022_latest=Path(candidate_panel_2022_latest),
            readiness=readiness,
            price_coverage=price_coverage,
            candidate_context=candidate_context,
            join_blockers=join_blockers,
        )

        price_coverage.to_csv(output / "twse_price_coverage_audit.csv", index=False, encoding="utf-8-sig")
        # Compatibility with Research handoff naming.
        price_coverage.to_csv(
            output / "twse_sector_proxy_price_coverage_by_ticker.csv", index=False, encoding="utf-8-sig"
        )
        price_blockers.to_csv(output / "twse_sector_proxy_price_blockers.csv", index=False, encoding="utf-8-sig")
        if candidate_context.empty:
            candidate_blockers.to_csv(output / "candidate_sector_context_blockers.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=_candidate_context_columns()).to_csv(
                output / "candidate_sector_context_2022_latest.csv", index=False, encoding="utf-8-sig"
            )
        else:
            candidate_context.to_csv(output / "candidate_sector_context_2022_latest.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=_candidate_blocker_columns()).to_csv(
                output / "candidate_sector_context_blockers.csv", index=False, encoding="utf-8-sig"
            )
        join_blockers.to_csv(output / "sector_proxy_join_blockers.csv", index=False, encoding="utf-8-sig")
        future_audit.to_csv(output / "future_data_violation_audit.csv", index=False, encoding="utf-8-sig")
        (output / "readiness_for_experiments.json").write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "sector_proxy_strategy_replay_readiness.json").write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "dynamic_pool1_sector_proxy_readiness_expansion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(readiness), encoding="utf-8")

        pd.DataFrame(
            [
                {
                    "step": TASK_ID,
                    "status": "completed_data_needed_readiness_expansion",
                    "output_dir": str(output),
                }
            ]
        ).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("_", ".")
    if text.endswith(".TW"):
        return text
    return f"{text}.TW"


def _ticker_stem(ticker: str) -> str:
    return _normalize_ticker(ticker).replace(".TW", "_TW")


def _discover_price_files(price_cache_dir: Path, price_source_registry: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    if price_cache_dir.exists():
        for path in price_cache_dir.rglob("*.csv"):
            ticker = _ticker_from_price_path(path)
            if ticker:
                files.setdefault(ticker, [])
                if path not in files[ticker]:
                    files[ticker].append(path)

    if price_source_registry.exists():
        try:
            registry = pd.read_csv(price_source_registry).fillna("")
        except Exception:
            registry = pd.DataFrame()
        if {"ticker", "source_path"}.issubset(registry.columns):
            for row in registry.to_dict(orient="records"):
                ticker = _normalize_ticker(row.get("ticker", ""))
                path = Path(str(row.get("source_path", "")))
                if ticker and path.exists():
                    files.setdefault(ticker, [])
                    if path not in files[ticker]:
                        files[ticker].append(path)
    return files


def _ticker_from_price_path(path: Path) -> str:
    stem = path.stem.upper().replace("_", ".")
    parts = stem.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "TW":
        return f"{parts[0]}.TW"
    if stem.endswith(".TW") and stem[:-3].isdigit():
        return stem
    return ""


def _build_price_coverage(sector_panel: pd.DataFrame, price_files: dict[str, list[Path]]) -> pd.DataFrame:
    tickers = _sector_proxy_tickers(sector_panel)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        paths = sorted(price_files.get(ticker, []), key=lambda p: (len(str(p)), str(p)))
        meta = _price_meta(paths)
        sector_dates = sector_panel[sector_panel["ticker_with_suffix"].astype(str).eq(ticker)]
        first_sector_date = _min_date_str(sector_dates.get("as_of_date", pd.Series(dtype=str)))
        last_sector_date = _max_date_str(sector_dates.get("as_of_date", pd.Series(dtype=str)))
        price_data_ready = bool(meta["row_count"] > 0)
        outcome_60d_ready = bool(price_data_ready and meta["last_price_date"] and meta["last_price_date"] >= last_sector_date)
        rows.append(
            {
                "ticker": ticker,
                "sector_proxy_rows": int(len(sector_dates)),
                "first_sector_as_of_date": first_sector_date,
                "last_sector_as_of_date": last_sector_date,
                "price_data_ready": price_data_ready,
                "price_outcome_coverage_ready": outcome_60d_ready,
                "price_row_count": int(meta["row_count"]),
                "first_price_date": meta["first_price_date"],
                "last_price_date": meta["last_price_date"],
                "price_source_count": int(len(paths)),
                "primary_price_source_path": str(paths[0]) if paths else "",
                "price_source_paths": "|".join(str(path) for path in paths),
                "blocked_reason": "" if price_data_ready else "missing_local_price_cache_for_twse_sector_proxy_ticker",
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def _sector_proxy_tickers(sector_panel: pd.DataFrame) -> list[str]:
    if "ticker_with_suffix" in sector_panel.columns:
        raw = sector_panel["ticker_with_suffix"]
    elif "ticker" in sector_panel.columns:
        raw = sector_panel["ticker"]
    else:
        raw = pd.Series(dtype=str)
    return sorted({_normalize_ticker(value) for value in raw if _normalize_ticker(value)})


def _price_meta(paths: list[Path]) -> dict[str, Any]:
    first_dates: list[str] = []
    last_dates: list[str] = []
    row_count = 0
    for path in paths:
        try:
            frame = pd.read_csv(path, usecols=lambda column: str(column).lower() == "date")
        except Exception:
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
        if "date" not in [str(column).lower() for column in frame.columns]:
            continue
        date_column = next(column for column in frame.columns if str(column).lower() == "date")
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
        if dates.empty:
            continue
        row_count += int(len(dates))
        first_dates.append(str(dates.min()))
        last_dates.append(str(dates.max()))
    return {
        "row_count": row_count,
        "first_price_date": min(first_dates) if first_dates else "",
        "last_price_date": max(last_dates) if last_dates else "",
    }


def _min_date_str(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    return dates.min().strftime("%Y-%m-%d") if not dates.empty else ""


def _max_date_str(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    return dates.max().strftime("%Y-%m-%d") if not dates.empty else ""


def _build_price_blockers(price_coverage: pd.DataFrame) -> pd.DataFrame:
    blockers = price_coverage[~price_coverage["price_data_ready"].astype(bool)].copy()
    if blockers.empty:
        return pd.DataFrame(columns=_price_blocker_columns())
    blockers["blocker_type"] = "missing_price_outcome_coverage"
    blockers["required_action"] = "fill_twse_sector_proxy_ticker_price_cache_before_outcome_or_strategy_diagnostic"
    return blockers[_price_blocker_columns()]


def _build_candidate_sector_context_2022_latest(
    *,
    sector_panel: pd.DataFrame,
    candidate_panel_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_panel_path.exists():
        return pd.DataFrame(columns=_candidate_context_columns()), pd.DataFrame(
            [
                {
                    "blocker_type": "missing_2022_latest_candidate_panel",
                    "source_path": str(candidate_panel_path),
                    "blocked_reason": "No 2022+ Dynamic Pool1 candidate ranking panel exists in Core outputs.",
                    "required_action": "produce date-aware 2022-latest Dynamic Pool1 candidate panel before sector context expansion",
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    candidates = pd.read_csv(candidate_panel_path).fillna("")
    required = {"date", "candidate_ticker"}
    if not required.issubset(candidates.columns):
        return pd.DataFrame(columns=_candidate_context_columns()), pd.DataFrame(
            [
                {
                    "blocker_type": "candidate_panel_schema_missing_columns",
                    "source_path": str(candidate_panel_path),
                    "blocked_reason": f"Candidate panel must include {sorted(required)}.",
                    "required_action": "regenerate candidate panel with date and candidate_ticker columns",
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    candidates = candidates.copy()
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates = candidates[candidates["date"] >= pd.Timestamp("2022-01-01")]
    if candidates.empty:
        return pd.DataFrame(columns=_candidate_context_columns()), pd.DataFrame(
            [
                {
                    "blocker_type": "candidate_panel_has_no_2022_latest_rows",
                    "source_path": str(candidate_panel_path),
                    "blocked_reason": "Candidate panel exists but contains no rows on or after 2022-01-01.",
                    "required_action": "produce 2022-latest candidate sector context input",
                    "diagnostic_only": True,
                    "active_in_trade_decision": False,
                }
            ]
        )
    for column in ["candidate_name", "candidate_source"]:
        if column not in candidates.columns:
            candidates[column] = ""
    candidates["date"] = candidates["date"].dt.strftime("%Y-%m-%d")
    candidates["candidate_ticker"] = candidates["candidate_ticker"].map(_normalize_ticker)

    anchors = sector_panel.copy()
    anchors["as_of_date"] = pd.to_datetime(anchors["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    anchors["ticker_with_suffix"] = anchors["ticker_with_suffix"].map(_normalize_ticker)
    merged = candidates.merge(
        anchors[
            [
                "as_of_date",
                "ticker_with_suffix",
                "sector_code",
                "sector_name",
                "source_date",
                "effective_date",
                "membership_policy",
                "daily_exact",
                "twse_only",
                "tpex_included",
                "mainline_theme_ready",
                "diagnostic_only",
                "active_in_trade_decision",
            ]
        ],
        left_on=["date", "candidate_ticker"],
        right_on=["as_of_date", "ticker_with_suffix"],
        how="left",
    )
    sector_code_ready = merged["sector_code"].fillna("").astype(str).ne("")
    merged["context_status"] = merged["sector_code"].where(
        sector_code_ready,
        "missing_twse_monthly_anchor_join",
    )
    merged["context_status"] = merged["context_status"].where(
        merged["context_status"].eq("missing_twse_monthly_anchor_join"),
        "twse_monthly_anchor_exact_date_match",
    )
    merged["sector_source_date"] = merged["source_date"]
    merged["sector_effective_date"] = merged["effective_date"]
    merged["sector_as_of_date"] = merged["as_of_date"]
    return merged[_candidate_context_columns()].sort_values(["date", "candidate_ticker"]).reset_index(drop=True), pd.DataFrame(
        columns=_candidate_blocker_columns()
    )


def _join_blockers(price_blockers: pd.DataFrame, candidate_blockers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in price_blockers.to_dict(orient="records"):
        rows.append(
            {
                "blocker_area": "price_outcome_coverage",
                "ticker": item.get("ticker", ""),
                "blocker_type": item.get("blocker_type", ""),
                "blocked_reason": item.get("blocked_reason", ""),
                "required_action": item.get("required_action", ""),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    for item in candidate_blockers.to_dict(orient="records"):
        rows.append(
            {
                "blocker_area": "candidate_sector_context_2022_latest",
                "ticker": "",
                "blocker_type": item.get("blocker_type", ""),
                "blocked_reason": item.get("blocked_reason", ""),
                "required_action": item.get("required_action", ""),
                "diagnostic_only": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _readiness_for_experiments(
    *,
    sector_readiness: dict[str, Any],
    sector_panel: pd.DataFrame,
    price_coverage: pd.DataFrame,
    candidate_context: pd.DataFrame,
    candidate_blockers: pd.DataFrame,
) -> dict[str, Any]:
    total = int(len(price_coverage))
    ready = int(price_coverage["price_data_ready"].astype(bool).sum()) if total else 0
    coverage = round(ready / total, 6) if total else 0.0
    candidate_ready = bool(not candidate_context.empty and candidate_blockers.empty)
    data_needed = bool(coverage < 0.95 or not candidate_ready)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "data_needed" if data_needed else "ready_for_more_complete_twse_only_diagnostic",
        "twse_only_sector_proxy_readiness_expanded": True,
        "ready_for_experiments_more_complete_diagnostic": bool(not data_needed),
        "ready_for_strategy_replay": False,
        "strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "diagnostic_only": True,
        "twse_only": True,
        "tpex_included": False,
        "mainline_theme_ready": False,
        "daily_exact": False,
        "official_sector_proxy_not_ai_mainline_theme": True,
        "price_coverage_total_tickers": total,
        "price_coverage_ready_tickers": ready,
        "price_coverage_missing_tickers": int(total - ready),
        "price_coverage_ratio": coverage,
        "candidate_sector_context_2022_latest_ready": candidate_ready,
        "candidate_sector_context_2022_latest_rows": int(len(candidate_context)),
        "candidate_sector_context_2022_latest_blockers": int(len(candidate_blockers)),
        "sector_proxy_rows": int(len(sector_panel)),
        "sector_proxy_unique_tickers": int(total),
        "source_sector_proxy_status": sector_readiness.get("status", ""),
        "future_data_violation_count": int(sector_readiness.get("future_data_violation_count", 0) or 0),
        "remaining_blockers": _remaining_blockers(coverage, candidate_ready),
    }


def _remaining_blockers(price_coverage_ratio: float, candidate_ready: bool) -> list[str]:
    blockers: list[str] = []
    if price_coverage_ratio < 0.95:
        blockers.append("full_twse_price_outcome_coverage_still_below_readiness_threshold")
    if not candidate_ready:
        blockers.append("missing_2022_latest_candidate_sector_context")
    blockers.extend(
        [
            "tpex_sector_membership_not_included_twse_only_scope",
            "ai_mainline_theme_taxonomy_not_ready",
            "monthly_anchor_sector_membership_is_not_daily_exact",
        ]
    )
    return blockers


def _future_data_violation_audit(sector_readiness: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_area": "twse_sector_proxy_readiness_expansion",
                "future_data_violation": False,
                "future_data_violation_count": int(sector_readiness.get("future_data_violation_count", 0) or 0),
                "audit_reason": "This package audits local price coverage and joins PIT monthly-anchor sector context; it does not use forward return as a live rule.",
            }
        ]
    )


def _manifest(
    *,
    output: Path,
    sector_proxy_output: Path,
    price_cache_dir: Path,
    candidate_panel_2022_latest: Path,
    readiness: dict[str, Any],
    price_coverage: pd.DataFrame,
    candidate_context: pd.DataFrame,
    join_blockers: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": readiness["status"],
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "sector_proxy_output": str(sector_proxy_output),
        "price_cache_dir": str(price_cache_dir),
        "candidate_panel_2022_latest": str(candidate_panel_2022_latest),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "strategy_replay": False,
        "diagnostic_only": True,
        "twse_only": True,
        "tpex_included": False,
        "price_coverage_total_tickers": readiness["price_coverage_total_tickers"],
        "price_coverage_ready_tickers": readiness["price_coverage_ready_tickers"],
        "price_coverage_ratio": readiness["price_coverage_ratio"],
        "candidate_context_rows": int(len(candidate_context)),
        "join_blocker_rows": int(len(join_blockers)),
        "future_data_violation_count": readiness["future_data_violation_count"],
        "outputs": {
            "manifest": "manifest.json",
            "dynamic_pool1_sector_proxy_readiness_expansion_manifest": "dynamic_pool1_sector_proxy_readiness_expansion_manifest.json",
            "twse_price_coverage_audit": "twse_price_coverage_audit.csv",
            "twse_sector_proxy_price_coverage_by_ticker": "twse_sector_proxy_price_coverage_by_ticker.csv",
            "twse_sector_proxy_price_blockers": "twse_sector_proxy_price_blockers.csv",
            "candidate_sector_context_2022_latest": "candidate_sector_context_2022_latest.csv",
            "candidate_sector_context_blockers": "candidate_sector_context_blockers.csv",
            "sector_proxy_join_blockers": "sector_proxy_join_blockers.csv",
            "readiness_for_experiments": "readiness_for_experiments.json",
            "sector_proxy_strategy_replay_readiness": "sector_proxy_strategy_replay_readiness.json",
            "future_data_violation_audit": "future_data_violation_audit.csv",
        },
    }


def _summary_zh(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 sector proxy readiness expansion",
            "",
            f"- 狀態：{readiness['status']}",
            f"- TWSE sector proxy tickers：{readiness['price_coverage_total_tickers']}",
            f"- 本地價格可接：{readiness['price_coverage_ready_tickers']} / {readiness['price_coverage_total_tickers']}，coverage={readiness['price_coverage_ratio']:.2%}",
            f"- 2022+ candidate sector context ready：{readiness['candidate_sector_context_2022_latest_ready']}",
            "- 邊界：TWSE-only、official sector proxy、diagnostic-only；不是 full-market、不是 AI/mainline/theme、不是 strategy replay。",
            "- formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false。",
            "",
            "## 下一步",
            "- 先補 full TWSE price outcome coverage；同時產出 2022-latest Dynamic Pool1 candidate panel，才能做更完整 sector proxy diagnostic。",
        ]
    )


def _price_blocker_columns() -> list[str]:
    return [
        "ticker",
        "sector_proxy_rows",
        "first_sector_as_of_date",
        "last_sector_as_of_date",
        "blocker_type",
        "blocked_reason",
        "required_action",
        "diagnostic_only",
        "active_in_trade_decision",
    ]


def _candidate_context_columns() -> list[str]:
    return [
        "date",
        "candidate_ticker",
        "candidate_name",
        "candidate_source",
        "sector_code",
        "sector_name",
        "sector_source_date",
        "sector_effective_date",
        "sector_as_of_date",
        "membership_policy",
        "daily_exact",
        "twse_only",
        "tpex_included",
        "mainline_theme_ready",
        "diagnostic_only",
        "active_in_trade_decision",
        "context_status",
    ]


def _candidate_blocker_columns() -> list[str]:
    return [
        "blocker_type",
        "source_path",
        "blocked_reason",
        "required_action",
        "diagnostic_only",
        "active_in_trade_decision",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand TWSE-only Dynamic Pool1 sector proxy readiness.")
    parser.add_argument("--sector-proxy-output", default=DEFAULT_SECTOR_PROXY_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--candidate-panel-2022-latest", default=DEFAULT_CANDIDATE_PANEL_2022_LATEST)
    args = parser.parse_args(argv)
    run_dynamic_pool1_sector_proxy_readiness_expansion(
        sector_proxy_output=args.sector_proxy_output,
        output_dir=args.output_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        candidate_panel_2022_latest=args.candidate_panel_2022_latest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
