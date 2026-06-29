from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
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


TASK_ID = "TASK-BACKTEST-CORE-SUPPLEMENTAL-PRICE-SOURCE-COVERAGE-ADOPTION-PHASE6-20260629"
DEFAULT_OUTPUT_DIR = "outputs/core_data_readiness_with_supplemental_price_phase6_20260629"


def run_data_readiness_with_supplemental(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    price_roots: tuple[str | Path, ...] = DEFAULT_PRICE_ROOTS,
    registry_path: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
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
        log("load_constituents", "started", str(constituents_path))
        constituents = _load_constituents(Path(constituents_path))
        tickers = sorted(set(BENCHMARK_TICKERS) | set(constituents["ticker"].dropna().astype(str)))

        log("scan_base_price_cache", "started", ",".join(str(root) for root in price_roots))
        price_sources = _scan_price_sources(tuple(Path(root) for root in price_roots))
        base_coverage = _price_coverage_matrix(tickers, price_sources)

        log("load_supplemental_registry", "started", str(registry_path))
        supplemental_registry = load_price_source_registry(registry_path)
        supplemental_usage = _supplemental_source_usage(supplemental_registry)
        combined = _combined_price_coverage(tickers, base_coverage, supplemental_usage)
        readiness = _readiness_ledger(combined, constituents)
        blockers = _remaining_blockers(combined)
        manifest = _manifest(output, registry_path, combined, supplemental_usage, blockers)

        log("write_outputs", "started", "")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        combined.to_csv(output / "price_coverage_with_supplemental.csv", index=False, encoding="utf-8-sig")
        supplemental_usage.to_csv(output / "supplemental_source_usage.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "readiness_ledger.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "remaining_blockers.csv", index=False, encoding="utf-8-sig")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, combined, blockers), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_data_readiness_with_supplemental", "error": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _supplemental_source_usage(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in registry.to_dict(orient="records"):
        if not _as_bool(row.get("price_source_ready")):
            continue
        source_path = Path(str(row.get("source_path", "")))
        if not source_path.exists():
            rows.append(
                {
                    **row,
                    "source_file_exists": False,
                    "loaded_first_date": "",
                    "loaded_last_date": "",
                    "loaded_row_count": 0,
                    "used_in_combined_coverage": False,
                    "usage_blocker": f"source file missing: {source_path}",
                }
            )
            continue
        frame = load_price_csv(source_path)
        rows.append(
            {
                **row,
                "source_file_exists": True,
                "loaded_first_date": frame.index.min().strftime("%Y-%m-%d") if not frame.empty else "",
                "loaded_last_date": frame.index.max().strftime("%Y-%m-%d") if not frame.empty else "",
                "loaded_row_count": int(len(frame)),
                "used_in_combined_coverage": not frame.empty and not _as_bool(row.get("synthetic_used")),
                "usage_blocker": "" if not _as_bool(row.get("synthetic_used")) else "synthetic source is not allowed",
            }
        )
    return pd.DataFrame(rows)


def _combined_price_coverage(tickers: list[str], base: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
    expected_days = pd.bdate_range(AUDIT_START, AUDIT_END)
    expected_first = expected_days.min()
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        base_row = _row_for_ticker(base, ticker)
        supplemental_rows = supplemental[
            supplemental.get("ticker", pd.Series(dtype=str)).astype(str).eq(ticker)
            & supplemental.get("used_in_combined_coverage", pd.Series(dtype=bool)).map(_as_bool)
        ]
        date_index = _date_index_from_base_row(base_row)
        supplemental_dates: list[pd.Timestamp] = []
        supplemental_source_types: list[str] = []
        supplemental_paths: list[str] = []
        supplemental_synthetic = False
        supplemental_strategy_ready = False
        for row in supplemental_rows.to_dict(orient="records"):
            frame = load_price_csv(row["source_path"])
            supplemental_dates.extend(frame.index.tolist())
            supplemental_source_types.append(str(row.get("source_type", "")))
            supplemental_paths.append(str(row.get("source_path", "")))
            supplemental_synthetic = supplemental_synthetic or _as_bool(row.get("synthetic_used"))
            supplemental_strategy_ready = supplemental_strategy_ready or _as_bool(row.get("strategy_ready"))
        if supplemental_dates:
            date_index = date_index.union(pd.DatetimeIndex(supplemental_dates))
        date_index = date_index.sort_values().unique()
        covered = expected_days.intersection(date_index)
        combined_first = date_index.min().strftime("%Y-%m-%d") if len(date_index) else ""
        combined_last = date_index.max().strftime("%Y-%m-%d") if len(date_index) else ""
        combined_ready = bool(len(date_index) and date_index.min() <= expected_first and date_index.max() >= pd.Timestamp(AUDIT_END))
        rows.append(
            {
                "ticker": ticker,
                "base_source": str(base_row.get("source", "")),
                "base_first_date": str(base_row.get("first_date", "")),
                "base_last_date": str(base_row.get("last_date", "")),
                "base_status": str(base_row.get("status", "missing")),
                "supplemental_source_path": ";".join(supplemental_paths),
                "supplemental_source_type": ";".join(sorted(set(filter(None, supplemental_source_types)))),
                "supplemental_first_date": min(supplemental_dates).strftime("%Y-%m-%d") if supplemental_dates else "",
                "supplemental_last_date": max(supplemental_dates).strftime("%Y-%m-%d") if supplemental_dates else "",
                "supplemental_synthetic_used": supplemental_synthetic,
                "combined_first_date": combined_first,
                "combined_last_date": combined_last,
                "combined_row_count": int(len(date_index)),
                "combined_missing_bday_count": int(len(expected_days) - len(covered)),
                "combined_coverage_ratio": round(len(covered) / len(expected_days), 6) if len(expected_days) else 0.0,
                "price_source_ready": combined_ready and not supplemental_synthetic,
                "strategy_ready": False,
                "strategy_ready_blocker": "TW50 PIT constituents, 2014-2021 formal target stream, and execution ledger remain incomplete.",
                "provenance": "base_cache_only" if not supplemental_paths else "base_cache_plus_supplemental_registry",
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def _row_for_ticker(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    row = frame[frame["ticker"].astype(str).eq(ticker)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _date_index_from_base_row(row: dict[str, Any]) -> pd.DatetimeIndex:
    source = str(row.get("source", ""))
    if source and Path(source).exists():
        try:
            return load_price_csv(source).index
        except Exception:
            return pd.DatetimeIndex([])
    first = str(row.get("first_date", ""))
    last = str(row.get("last_date", ""))
    if first and last:
        return pd.bdate_range(first, last)
    return pd.DatetimeIndex([])


def _readiness_ledger(combined: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    row_00631l = _row_for_ticker(combined, "00631L.TW")
    price_ready_count = int(combined["price_source_ready"].sum()) if not combined.empty else 0
    total = int(len(combined))
    return pd.DataFrame(
        [
            {
                "layer": "00631l_price_coverage",
                "status": "price_source_ready" if row_00631l.get("price_source_ready") else "not_ready",
                "price_source_ready": bool(row_00631l.get("price_source_ready", False)),
                "strategy_ready": False,
                "detail": f"base_first={row_00631l.get('base_first_date', '')}; supplemental_first={row_00631l.get('supplemental_first_date', '')}; combined_first={row_00631l.get('combined_first_date', '')}.",
            },
            {
                "layer": "provisional_universe_price_coverage",
                "status": "partial_price_coverage",
                "price_source_ready": price_ready_count == total and total > 0,
                "strategy_ready": False,
                "detail": f"{price_ready_count}/{total} provisional tickers are price-source ready. Current/provisional universe has {constituents['ticker'].nunique() if not constituents.empty else 0} current TW50 tickers and is not complete historical PIT.",
            },
            {
                "layer": "tw50_exact_pit_archive",
                "status": "blocked_missing_exact_archive",
                "price_source_ready": False,
                "strategy_ready": False,
                "detail": "Exact/source-backed PIT constituents for 2014/11-2023/12 are still missing.",
            },
            {
                "layer": "formal_target_signal_stream",
                "status": "blocked_missing_2014_2021_target_stream",
                "price_source_ready": False,
                "strategy_ready": False,
                "detail": "Price coverage alone cannot rebuild formal target stream; Pool1 ranking and Pool2 confirmation evidence are still needed.",
            },
        ]
    )


def _remaining_blockers(combined: pd.DataFrame) -> pd.DataFrame:
    row_00631l = _row_for_ticker(combined, "00631L.TW")
    rows = [
        {
            "blocker": "tw50_exact_pit_archive",
            "severity": "blocking_strategy_replay",
            "detail": "Still cannot use current/provisional universe as complete historical TW50 universe.",
            "blocks_strategy_ready": True,
        },
        {
            "blocker": "formal_target_signal_stream_2014_2021",
            "severity": "blocking_strategy_replay",
            "detail": "Need Pool1 ranking/score margin and Pool2 confirmation replay contract for 2014-2021.",
            "blocks_strategy_ready": True,
        },
        {
            "blocker": "execution_ledger_2014_2021",
            "severity": "blocking_strategy_replay",
            "detail": "Execution replay can only run after target stream exists.",
            "blocks_strategy_ready": True,
        },
        {
            "blocker": "00631l_adjusted_close_distribution_policy",
            "severity": "price_source_caveat",
            "detail": f"00631L supplemental source now starts {row_00631l.get('combined_first_date', '')}, but final total-return/adjustment policy still needs validation.",
            "blocks_strategy_ready": True,
        },
    ]
    return pd.DataFrame(rows)


def _manifest(output: Path, registry_path: str | Path, combined: pd.DataFrame, supplemental: pd.DataFrame, blockers: pd.DataFrame) -> dict[str, Any]:
    row_00631l = _row_for_ticker(combined, "00631L.TW")
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "registry_path": str(Path(registry_path).as_posix()),
        "supplemental_source_count": int(len(supplemental)),
        "supplemental_sources_used_count": int(supplemental["used_in_combined_coverage"].map(_as_bool).sum()) if not supplemental.empty else 0,
        "00631l_base_first_date": row_00631l.get("base_first_date", ""),
        "00631l_supplemental_first_date": row_00631l.get("supplemental_first_date", ""),
        "00631l_combined_first_date": row_00631l.get("combined_first_date", ""),
        "00631l_combined_last_date": row_00631l.get("combined_last_date", ""),
        "00631l_source_type": row_00631l.get("supplemental_source_type", ""),
        "00631l_synthetic_used": bool(row_00631l.get("supplemental_synthetic_used", False)),
        "00631l_price_source_ready": bool(row_00631l.get("price_source_ready", False)),
        "strategy_ready": False,
        "remaining_blocker_count": int(len(blockers)),
        "output_dir": str(output.resolve()),
    }


def _summary_zh(manifest: dict[str, Any], combined: pd.DataFrame, blockers: pd.DataFrame) -> str:
    row_00631l = _row_for_ticker(combined, "00631L.TW")
    lines = [
        "# Data Readiness with Supplemental Price Source Phase 6",
        "",
        "## 結論",
        "",
        "- coverage/readiness 已採用 supplemental price source registry。",
        f"- 00631L base/local cache first_date：{row_00631l.get('base_first_date', '')}。",
        f"- 00631L supplemental first_date：{row_00631l.get('supplemental_first_date', '')}。",
        f"- 00631L combined first_date：{row_00631l.get('combined_first_date', '')}。",
        f"- source_type：{row_00631l.get('supplemental_source_type', '')}；synthetic_used={row_00631l.get('supplemental_synthetic_used', False)}。",
        "- 00631L price gap 已收斂到 coverage/readiness 層可見，但 strategy_ready 仍為 false。",
        "",
        "## 剩餘 blocker",
        "",
    ]
    for row in blockers.to_dict(orient="records"):
        lines.append(f"- `{row['blocker']}`：{row['detail']}")
    return "\n".join(lines) + "\n"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data readiness coverage that adopts supplemental price source registry.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--price-root", action="append", default=[])
    parser.add_argument("--registry-path", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    price_roots = tuple(args.price_root) if args.price_root else DEFAULT_PRICE_ROOTS
    output = run_data_readiness_with_supplemental(
        constituents_path=args.constituents_path,
        price_roots=tuple(Path(root) for root in price_roots),
        registry_path=args.registry_path,
        output_dir=args.output_dir,
    )
    print(f"DATA_READINESS_WITH_SUPPLEMENTAL_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
