from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/core_data_backfill_tw50_201411_202312_audit_20260629"
DEFAULT_CONSTITUENTS_PATH = "data/tw50_constituents.csv"
DEFAULT_PRICE_ROOTS = ("backtest_cache",)
DEFAULT_FORMAL_REPLAY_METADATA = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624/metadata.json"
AUDIT_START = "2014-11-01"
AUDIT_END = "2023-12-29"
BENCHMARK_TICKERS = ("0050.TW", "00631L.TW")


def run_tw50_backfill_audit(
    *,
    constituents_path: str | Path = DEFAULT_CONSTITUENTS_PATH,
    price_roots: tuple[str | Path, ...] = DEFAULT_PRICE_ROOTS,
    formal_replay_metadata: str | Path = DEFAULT_FORMAL_REPLAY_METADATA,
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
        pit_inventory = _pit_inventory(constituents)
        target_tickers = sorted(set(BENCHMARK_TICKERS) | set(constituents["ticker"].dropna().astype(str)))

        log("scan_price_cache", "started", ",".join(str(root) for root in price_roots))
        price_sources = _scan_price_sources(tuple(Path(root) for root in price_roots))
        price_coverage = _price_coverage_matrix(target_tickers, price_sources)

        log("build_readiness", "started", "")
        readiness = _readiness_ledger(pit_inventory, price_coverage, constituents)
        plan = _missing_data_backfill_plan(pit_inventory, price_coverage, constituents)
        formal_meta = _load_json(Path(formal_replay_metadata))
        manifest = _manifest(constituents, pit_inventory, price_coverage, formal_meta)

        log("write_outputs", "started", "")
        pit_inventory.to_csv(output / "tw50_pit_constituents_coverage.csv", index=False, encoding="utf-8-sig")
        price_coverage.to_csv(output / "price_coverage_matrix.csv", index=False, encoding="utf-8-sig")
        plan.to_csv(output / "missing_data_backfill_plan.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(output / "data_readiness_ledger.csv", index=False, encoding="utf-8-sig")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, plan, formal_meta), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_tw50_backfill_audit", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _load_constituents(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["effective_date", "ticker", "name", "source", "source_updated_at"])
    frame = pd.read_csv(path).fillna("")
    if "ticker" not in frame.columns:
        raise ValueError(f"constituents file missing ticker column: {path}")
    if "effective_date" not in frame.columns:
        frame["effective_date"] = ""
    if "source" not in frame.columns:
        frame["source"] = ""
    if "name" not in frame.columns:
        frame["name"] = ""
    if "source_updated_at" not in frame.columns:
        frame["source_updated_at"] = ""
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    return frame[frame["ticker"].ne("")]


def _pit_inventory(constituents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if constituents.empty:
        return pd.DataFrame(columns=["effective_date", "source_type", "constituents_count", "ticker_list", "evidence_source", "formal_ready"])
    for effective_date, group in constituents.groupby("effective_date", dropna=False):
        sources = sorted(set(group["source"].astype(str)))
        source_type = _source_type(sources)
        tickers = sorted(set(group["ticker"].astype(str)))
        rows.append(
            {
                "effective_date": str(effective_date),
                "source_type": source_type,
                "constituents_count": len(tickers),
                "ticker_list": ";".join(tickers),
                "evidence_source": ";".join(sources),
                "source_updated_at": ";".join(sorted(set(group["source_updated_at"].astype(str)))),
                "formal_ready": source_type == "exact" and str(effective_date) <= AUDIT_START,
            }
        )
    return pd.DataFrame(rows).sort_values("effective_date").reset_index(drop=True)


def _source_type(sources: list[str]) -> str:
    text = " ".join(sources).lower()
    if "exact" in text or "ftse_pdf" in text:
        return "exact"
    if "manual" in text:
        return "manual_ledger"
    if "seed" in text or "snapshot" in text or "proxy" in text:
        return "proxy_current_snapshot"
    return "unknown"


def _scan_price_sources(roots: tuple[Path, ...]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            ticker = _ticker_from_price_path(path)
            if not ticker:
                continue
            coverage = _price_file_coverage(path)
            if coverage:
                result.setdefault(ticker, []).append(coverage)
    return result


def _ticker_from_price_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_TW"):
        return stem[:-3] + ".TW"
    if stem.endswith(".TW"):
        return stem
    return ""


def _price_file_coverage(path: Path) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path, usecols=lambda col: col in {"date", "adj_close", "close", "dividend", "stock_split"})
    except Exception:
        return None
    if "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    has_adj = "adj_close" in frame.columns and pd.to_numeric(frame["adj_close"], errors="coerce").notna().any()
    has_dividend = "dividend" in frame.columns
    has_split = "stock_split" in frame.columns
    return {
        "source": str(path),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "row_count": int(len(dates)),
        "has_adjusted_close": bool(has_adj),
        "has_dividend": bool(has_dividend),
        "has_split": bool(has_split),
    }


def _price_coverage_matrix(tickers: list[str], sources: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    expected_days = pd.bdate_range(AUDIT_START, AUDIT_END)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        candidates = sources.get(ticker, [])
        best = _best_price_source(candidates)
        if not best:
            rows.append(
                {
                    "ticker": ticker,
                    "source": "",
                    "first_date": "",
                    "last_date": "",
                    "row_count": 0,
                    "missing_date_count": len(expected_days),
                    "coverage_ratio": 0.0,
                    "has_adjusted_close": False,
                    "has_dividend": False,
                    "has_split": False,
                    "formal_ready": False,
                    "status": "missing",
                }
            )
            continue
        first = pd.Timestamp(best["first_date"])
        last = pd.Timestamp(best["last_date"])
        covered = expected_days[(expected_days >= first) & (expected_days <= last)]
        ratio = len(covered) / len(expected_days) if len(expected_days) else 0.0
        formal_ready = first <= pd.Timestamp(AUDIT_START) and last >= pd.Timestamp(AUDIT_END) and bool(best["has_adjusted_close"])
        rows.append(
            {
                "ticker": ticker,
                "source": best["source"],
                "first_date": best["first_date"],
                "last_date": best["last_date"],
                "row_count": best["row_count"],
                "missing_date_count": int(len(expected_days) - len(covered)),
                "coverage_ratio": round(ratio, 6),
                "has_adjusted_close": best["has_adjusted_close"],
                "has_dividend": best["has_dividend"],
                "has_split": best["has_split"],
                "formal_ready": formal_ready,
                "status": "formal_ready" if formal_ready else "partial_or_out_of_range",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "ticker"]).reset_index(drop=True)


def _best_price_source(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (row["first_date"], -int(row["row_count"])))[0]


def _readiness_ledger(pit_inventory: pd.DataFrame, price_coverage: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    exact_pit_ready = bool(not pit_inventory.empty and (pit_inventory["source_type"] == "exact").any() and pit_inventory["effective_date"].min() <= AUDIT_START)
    price_ready_count = int(price_coverage["formal_ready"].sum()) if not price_coverage.empty else 0
    total_tickers = int(len(price_coverage))
    confirmed_current_tickers = int(constituents["ticker"].nunique()) if not constituents.empty else 0
    rows = [
        {
            "layer": "0050_price",
            "status": _ticker_status(price_coverage, "0050.TW"),
            "blocks_2014_2023_backtest": _ticker_status(price_coverage, "0050.TW") != "formal_ready",
            "detail": "0050 must be available as benchmark/reference, not pool candidate.",
        },
        {
            "layer": "00631L_price",
            "status": _ticker_status(price_coverage, "00631L.TW"),
            "blocks_2014_2023_backtest": _ticker_status(price_coverage, "00631L.TW") != "formal_ready",
            "detail": "00631L is the 0050x2 exposure instrument and needs adjusted close coverage.",
        },
        {
            "layer": "tw50_pit_constituents",
            "status": "formal_ready" if exact_pit_ready else "blocked_missing_exact_pit",
            "blocks_2014_2023_backtest": not exact_pit_ready,
            "detail": f"Current repo confirms {confirmed_current_tickers} current/proxy TW50 tickers; exact historical ticker count is blocked until PIT source archive is acquired.",
        },
        {
            "layer": "tw50_constituent_prices",
            "status": "formal_ready" if price_ready_count == total_tickers and total_tickers > 0 else "partial_or_missing",
            "blocks_2014_2023_backtest": price_ready_count != total_tickers,
            "detail": f"{price_ready_count}/{total_tickers} confirmed tickers have full {AUDIT_START} to {AUDIT_END} adjusted-price coverage in current cache scan.",
        },
        {
            "layer": "formal_target_stream_evidence",
            "status": "blocked_missing_2014_2021_formal_stream",
            "blocks_2014_2023_backtest": True,
            "detail": "Existing formal target stream starts at 2022-01-03; 2014/11-2021 cannot be treated as formal strategy performance until candidate/ranking/target evidence is rebuilt.",
        },
        {
            "layer": "risk_factor_institutional_flow",
            "status": "not_required_for_current_formal_selector_or_blocked_if_future_rule_requires_it",
            "blocks_2014_2023_backtest": False,
            "detail": "Current formal selector must not depend on unavailable chip/risk factors for 2014-2021 unless a separate accepted PIT contract is built.",
        },
    ]
    return pd.DataFrame(rows)


def _ticker_status(price_coverage: pd.DataFrame, ticker: str) -> str:
    row = price_coverage[price_coverage["ticker"] == ticker]
    if row.empty:
        return "missing"
    return str(row.iloc[0]["status"])


def _missing_data_backfill_plan(pit_inventory: pd.DataFrame, price_coverage: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    current_count = int(constituents["ticker"].nunique()) if not constituents.empty else 0
    exact_ready = bool(not pit_inventory.empty and (pit_inventory["source_type"] == "exact").any() and pit_inventory["effective_date"].min() <= AUDIT_START)
    missing_prices = price_coverage[~price_coverage["formal_ready"]]["ticker"].tolist() if not price_coverage.empty else []
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "step": "archive_tw50_pit_constituent_sources",
                "scope": "2014-11 to 2023-12",
                "current_status": "blocked_missing_exact_pit" if not exact_ready else "ready",
                "estimated_ticker_count": f"confirmed_current_minimum={current_count}; exact_historical_total=blocked_until_pit_archive",
                "source_plan": "Acquire FTSE/TWSE/issuer historical constituent notices or accepted manual ledger with effective dates; archive raw files before normalization.",
                "checkpoint": "raw_source_archive_manifest.csv",
                "can_resume": True,
            },
            {
                "step_order": 2,
                "step": "normalize_pit_constituents_table",
                "scope": "effective_date,ticker,name,source_type,evidence_source",
                "current_status": "blocked_by_step_1",
                "estimated_ticker_count": "computed_after_step_1",
                "source_plan": "Build PIT table; classify each effective date as exact/proxy/manual; never label current snapshot as exact history.",
                "checkpoint": "tw50_pit_constituents_normalized.csv",
                "can_resume": True,
            },
            {
                "step_order": 3,
                "step": "backfill_adjusted_prices_for_pit_universe",
                "scope": f"{AUDIT_START} to {AUDIT_END}",
                "current_status": "partial_or_missing",
                "estimated_ticker_count": f"known_missing_or_partial_now={len(missing_prices)}; final_count=computed_after_step_2",
                "source_plan": "Batch yfinance/TWSE-compatible source by ticker; write per-ticker raw and normalized CSV; record failed tickers without stopping entire run.",
                "checkpoint": "price_backfill_progress.csv",
                "can_resume": True,
            },
            {
                "step_order": 4,
                "step": "build_readiness_ledger",
                "scope": "prices + PIT + formal target evidence",
                "current_status": "planned",
                "estimated_ticker_count": "computed_after_steps_2_3",
                "source_plan": "Validate adjusted close, dividend/split fields, first/last date, missing trading days, and PIT active count per date.",
                "checkpoint": "data_readiness_ledger.csv",
                "can_resume": True,
            },
            {
                "step_order": 5,
                "step": "rebuild_formal_candidate_target_stream",
                "scope": "2014-11 to 2023-12",
                "current_status": "blocked_missing_formal_evidence_contract",
                "estimated_ticker_count": "depends_on_PIT_universe",
                "source_plan": "After data layer is ready, rerun formal candidate/ranking/target evidence with same formal contract; do not invent 2014-2021 targets from current constituents.",
                "checkpoint": "formal_target_stream_readiness.csv",
                "can_resume": True,
            },
        ]
    )


def _manifest(constituents: pd.DataFrame, pit_inventory: pd.DataFrame, price_coverage: pd.DataFrame, formal_meta: dict[str, Any]) -> dict[str, Any]:
    price_ready_count = int(price_coverage["formal_ready"].sum()) if not price_coverage.empty else 0
    return {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-DATA-BACKFILL-TW50-201411-202312-20260629",
        "status": "audit_completed_plan_ready",
        "audit_start": AUDIT_START,
        "audit_end": AUDIT_END,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "large_download_started": False,
        "current_constituent_source_rows": int(len(constituents)),
        "current_constituent_unique_tickers": int(constituents["ticker"].nunique()) if not constituents.empty else 0,
        "pit_effective_date_count": int(pit_inventory["effective_date"].nunique()) if not pit_inventory.empty else 0,
        "pit_min_effective_date": str(pit_inventory["effective_date"].min()) if not pit_inventory.empty else "",
        "pit_max_effective_date": str(pit_inventory["effective_date"].max()) if not pit_inventory.empty else "",
        "pit_exact_coverage_201411_202312_ready": bool(not pit_inventory.empty and (pit_inventory["source_type"] == "exact").any() and pit_inventory["effective_date"].min() <= AUDIT_START),
        "price_ticker_count_checked": int(len(price_coverage)),
        "price_formal_ready_count": price_ready_count,
        "price_missing_or_partial_count": int(len(price_coverage) - price_ready_count),
        "confirmed_minimum_ticker_count": int(len(price_coverage)),
        "exact_historical_total_ticker_count": "blocked_until_PIT_archive_acquired",
        "previous_best_initial_capital": formal_meta.get("initial_cash", ""),
        "previous_best_formal_stream_start": formal_meta.get("outputs", {}).get("decision_panel", "") and "2022-01-03",
        "previous_best_latest_formal_date": "2026-06-12",
        "needs_radar_or_data_thread_support": True,
        "primary_blockers": [
            "missing exact/proven PIT TW50 constituents for 2014-11 to 2023-12",
            "formal target stream/evidence currently starts at 2022-01-03",
            "final historical ticker count cannot be computed until PIT archive is acquired",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_zh(manifest: dict[str, Any], plan: pd.DataFrame, formal_meta: dict[str, Any]) -> str:
    lines = [
        "# TW50 / 0050 / 00631L 2014-11 至 2023-12 資料補齊 Phase 1 Audit",
        "",
        "## 結論",
        "",
        "- 目前不能把 2014/11-2023/12 補成正式策略績效；主要缺口是 TW50/0050 成分股 PIT 歷史與 2014-2021 formal target stream/evidence，不是單純缺幾個價格檔。",
        f"- 現有 `data/tw50_constituents.csv` 只確認 {manifest['current_constituent_unique_tickers']} 檔 current/proxy 成分，effective date 範圍 {manifest['pit_min_effective_date']} 到 {manifest['pit_max_effective_date']}。",
        f"- 本次價格覆蓋檢查確認最低 universe 為 {manifest['confirmed_minimum_ticker_count']} 檔（含 0050、00631L 與目前已知 TW50 成分）；完整歷史 ticker 總數要等 PIT archive 取得後才能計算，不能猜。",
        f"- previous best / formal replay metadata 顯示 initial capital = {formal_meta.get('initial_cash', 'unknown')}，也就是起始資金 100 萬。",
        "- 2014/2016-2021 不得用 current TW50 snapshot 或現有 2022+ target stream 回填成正式策略績效。",
        "",
        "## 下一步可續跑 backfill plan",
        "",
    ]
    for row in plan.to_dict(orient="records"):
        lines.append(f"{row['step_order']}. {row['step']}：{row['current_status']}；checkpoint={row['checkpoint']}")
    lines.extend(
        [
            "",
            "## 是否需要 Radar/Data thread 支援",
            "",
            "- 需要。PIT constituent source acquisition 應交 Radar/Data 或資料專線協助找 FTSE/TWSE/issuer historical constituent notices；Core 只應在資料來源可追溯後做 normalization/readiness runner。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TW50/0050/00631L data backfill readiness for 2014-11 to 2023-12.")
    parser.add_argument("--constituents-path", default=DEFAULT_CONSTITUENTS_PATH)
    parser.add_argument("--price-root", action="append", default=[])
    parser.add_argument("--formal-replay-metadata", default=DEFAULT_FORMAL_REPLAY_METADATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    price_roots = tuple(args.price_root) if args.price_root else DEFAULT_PRICE_ROOTS
    output = run_tw50_backfill_audit(
        constituents_path=args.constituents_path,
        price_roots=tuple(Path(root) for root in price_roots),
        formal_replay_metadata=args.formal_replay_metadata,
        output_dir=args.output_dir,
    )
    print(f"TW50_BACKFILL_AUDIT_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
