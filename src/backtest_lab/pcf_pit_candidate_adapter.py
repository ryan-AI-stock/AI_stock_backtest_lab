from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_MONTHLY_ANCHOR_PATH = (
    "outputs/core_0050_pcf_daily_monthly_anchor_pit_candidate_201411_202312_20260629/"
    "tw50_0050_pcf_daily_monthly_anchor_candidate.csv"
)
DEFAULT_PRICE_COVERAGE_PATH = (
    "outputs/core_0050_constituent_price_backfill_201411_latest_20260629/price_coverage_matrix.csv"
)
DEFAULT_OUTPUT_DIR = "outputs/core_0050_pit_candidate_backtest_data_readiness_201411_202312_20260629"

REQUIRED_ANCHOR_COLUMNS = {
    "effective_month",
    "effective_date",
    "ticker",
    "name",
    "source_url",
    "raw_source_id",
    "source_type",
    "formal_exact",
}


def load_0050_pcf_monthly_anchor(path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"0050 PCF monthly anchor candidate not found: {source}")
    frame = pd.read_csv(source).fillna("")
    missing = REQUIRED_ANCHOR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"0050 PCF monthly anchor missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("0050 PCF monthly anchor candidate is empty.")

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["effective_month"] = frame["effective_month"].astype(str).str.strip()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    if frame["effective_date"].isna().any():
        raise ValueError("0050 PCF monthly anchor contains invalid effective_date values.")
    if frame["ticker"].eq("").any() or frame["name"].eq("").any():
        raise ValueError("0050 PCF monthly anchor contains blank ticker/name rows.")
    if not frame["formal_exact"].map(_as_bool_false).all():
        raise ValueError("0050 PCF monthly anchor must keep formal_exact=false for every row.")
    frame["formal_exact"] = "false"
    if (frame["source_type"].astype(str) != "source_backed_manual_candidate").any():
        raise ValueError("0050 PCF monthly anchor source_type must be source_backed_manual_candidate.")
    if "proxy_row_used" in frame.columns and frame["proxy_row_used"].map(_as_bool).any():
        raise ValueError("0050 PCF monthly anchor must not include proxy rows.")
    if "proxy_row_used" in frame.columns:
        frame["proxy_row_used"] = "false"
    return frame.sort_values(["effective_month", "ticker"]).reset_index(drop=True)


def resolve_0050_constituents_for_date(
    signal_date: str | pd.Timestamp,
    *,
    monthly_anchor_path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH,
    mode: str = "calendar_month",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve 0050 PIT candidate constituents for a date.

    mode="calendar_month" returns the anchor for the date's yyyy-mm.
    mode="pit_safe" returns the latest anchor whose effective_date is <= signal_date.
    """
    anchor = load_0050_pcf_monthly_anchor(monthly_anchor_path)
    target = pd.Timestamp(signal_date).normalize()

    if mode not in {"calendar_month", "pit_safe"}:
        raise ValueError("mode must be calendar_month or pit_safe")
    if mode == "calendar_month":
        month = target.strftime("%Y-%m")
        selected = anchor[anchor["effective_month"].eq(month)].copy()
    else:
        eligible_months = anchor[anchor["effective_date"] <= target]
        if eligible_months.empty:
            month = target.strftime("%Y-%m")
            selected = anchor[anchor["effective_month"].eq(month)].copy()
        else:
            month = str(eligible_months.sort_values("effective_date").iloc[-1]["effective_month"])
            selected = anchor[anchor["effective_month"].eq(month)].copy()

    if selected.empty:
        first_month = str(anchor["effective_month"].min())
        last_month = str(anchor["effective_month"].max())
        raise ValueError(f"No 0050 PCF monthly anchor for {target.date()} in mode={mode}; range={first_month}..{last_month}")

    anchor_date = pd.Timestamp(selected["effective_date"].iloc[0]).normalize()
    metadata = {
        "query_date": target.strftime("%Y-%m-%d"),
        "mode": mode,
        "effective_month": str(selected["effective_month"].iloc[0]),
        "anchor_effective_date": anchor_date.strftime("%Y-%m-%d"),
        "constituent_count": int(selected["ticker"].nunique()),
        "formal_exact": False,
        "source_type": "source_backed_manual_candidate",
        "proxy_row_used": False,
        "active_in_trade_decision": False,
        "anchor_after_query_date": bool(anchor_date > target),
        "pit_safe_for_query_date": bool(anchor_date <= target),
    }
    selected.insert(0, "query_date", target.strftime("%Y-%m-%d"))
    selected.insert(1, "resolution_mode", mode)
    selected["anchor_after_query_date"] = str(metadata["anchor_after_query_date"]).lower()
    selected["pit_safe_for_query_date"] = str(metadata["pit_safe_for_query_date"]).lower()
    return selected.reset_index(drop=True), metadata


def run_0050_pit_candidate_backtest_data_readiness(
    *,
    monthly_anchor_path: str | Path = DEFAULT_MONTHLY_ANCHOR_PATH,
    price_coverage_path: str | Path = DEFAULT_PRICE_COVERAGE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_dates: Iterable[str] = ("2014-11-03", "2014-11-28", "2016-03-15", "2021-12-30", "2023-12-29"),
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

    try:
        log("load_monthly_anchor", "started", str(monthly_anchor_path))
        anchor = load_0050_pcf_monthly_anchor(monthly_anchor_path)
        sample = _date_to_anchor_sample(sample_dates, monthly_anchor_path)
        blockers = _backtest_data_blockers(anchor)
        price_req = _price_coverage_requirements(anchor, Path(price_coverage_path))
        signal_req = _signal_stream_requirements()
        execution_req = _execution_ledger_requirements()
        reader_contract = _reader_contract()

        log("write_outputs", "started", str(output))
        reader_contract.to_csv(output / "pit_candidate_reader_contract.csv", index=False, encoding="utf-8-sig")
        sample.to_csv(output / "date_to_monthly_anchor_sample.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "backtest_data_blockers.csv", index=False, encoding="utf-8-sig")
        price_req.to_csv(output / "price_coverage_requirements.csv", index=False, encoding="utf-8-sig")
        signal_req.to_csv(output / "signal_stream_requirements.csv", index=False, encoding="utf-8-sig")
        execution_req.to_csv(output / "execution_ledger_requirements.csv", index=False, encoding="utf-8-sig")
        (output / "pit_candidate_adapter_plan.md").write_text(_adapter_plan(), encoding="utf-8")
        (output / "next_experiments_task.md").write_text(_next_experiments_task(), encoding="utf-8")

        manifest = _manifest(anchor, price_req, blockers, monthly_anchor_path)
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_summary_zh(manifest, blockers), encoding="utf-8")
        pd.DataFrame([{"step": "run_0050_pit_candidate_backtest_data_readiness", "status": "completed"}]).to_csv(
            output / "completed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_0050_pit_candidate_backtest_data_readiness", "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv", index=False, encoding="utf-8-sig"
        )
        log("failed", "failed", str(exc))
        raise


def _date_to_anchor_sample(sample_dates: Iterable[str], monthly_anchor_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date in sample_dates:
        for mode in ("calendar_month", "pit_safe"):
            _, meta = resolve_0050_constituents_for_date(date, monthly_anchor_path=monthly_anchor_path, mode=mode)
            rows.append(meta)
    return pd.DataFrame(rows)


def _backtest_data_blockers(anchor: pd.DataFrame) -> pd.DataFrame:
    months = anchor["effective_month"].nunique()
    return pd.DataFrame(
        [
            {
                "blocker": "0050_pcf_monthly_anchor_reader",
                "status": "ready",
                "blocks_2014_2023_backtest": False,
                "detail": f"Reader can resolve {months}/110 months from source-backed manual candidate anchors.",
                "next_owner": "Core/Experiments",
            },
            {
                "blocker": "formal_exact_pit_policy",
                "status": "not_exact_but_candidate_ready",
                "blocks_2014_2023_backtest": False,
                "detail": "PCF/Daily is source-backed manual candidate, not official exact PIT. Replay must disclose formal_exact=false.",
                "next_owner": "Research/Core",
            },
            {
                "blocker": "price_coverage_for_anchor_universe",
                "status": "needs_verification",
                "blocks_2014_2023_backtest": True,
                "detail": "Every monthly anchor ticker needs adjusted price coverage from its first required date through 2023-12.",
                "next_owner": "Core/Data",
            },
            {
                "blocker": "formal_target_signal_stream_2014_2021",
                "status": "missing",
                "blocks_2014_2023_backtest": True,
                "detail": "Pool1 ranking, Pool2 confirmation, score margin, and formal target stream remain missing for 2014-2021.",
                "next_owner": "Core/Research/Experiments",
            },
            {
                "blocker": "execution_ledger_2014_2021",
                "status": "missing",
                "blocks_2014_2023_backtest": True,
                "detail": "Same-day/next-day execution ledger and cost model must be rebuilt after target stream exists.",
                "next_owner": "Core/Experiments",
            },
        ]
    )


def _price_coverage_requirements(anchor: pd.DataFrame, price_coverage_path: Path) -> pd.DataFrame:
    tickers = sorted({f"{str(t).zfill(4)}.TW" for t in anchor["ticker"].astype(str)})
    coverage = pd.DataFrame()
    if price_coverage_path.exists():
        coverage = pd.read_csv(price_coverage_path).fillna("")
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        cov = coverage[coverage["ticker"].astype(str).eq(ticker)] if not coverage.empty and "ticker" in coverage.columns else pd.DataFrame()
        if cov.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "coverage_status": "missing_coverage_row",
                    "first_date": "",
                    "last_date": "",
                    "adjusted_close_available": "",
                    "ready_for_backtest_price_only": "false",
                    "requirement": "Need adjusted price coverage for 2014/11-2023/12 where ticker appears in monthly anchors.",
                }
            )
        else:
            row = cov.iloc[0].to_dict()
            first = str(row.get("first_date", ""))
            last = str(row.get("last_date", ""))
            adjusted = str(row.get("adjusted_close_available", "")).lower() == "true"
            ready = str(row.get("ready_for_backtest_price_only", "")).lower() == "true"
            rows.append(
                {
                    "ticker": ticker,
                    "coverage_status": "price_only_ready" if ready else "not_ready",
                    "first_date": first,
                    "last_date": last,
                    "adjusted_close_available": str(adjusted).lower(),
                    "ready_for_backtest_price_only": str(ready).lower(),
                    "requirement": "Price-only readiness is necessary but not sufficient for strategy replay.",
                }
            )
    return pd.DataFrame(rows)


def _signal_stream_requirements() -> pd.DataFrame:
    rows = [
        ("pool1_candidate_ranking", "Pool1", "candidate_rank/candidate_score/score_margin", "missing_for_2014_2021"),
        ("pool2_confirmation_state", "Pool2", "confirmation/disagreement/risk state", "missing_for_2014_2021"),
        ("formal_target_selection_contract", "selector", "formal_target and target_formed_reason", "missing_for_2014_2021"),
        ("previous_target_contract", "report/execution", "previous formal target per trading day", "missing_for_2014_2021"),
    ]
    return pd.DataFrame(
        [
            {
                "requirement": req,
                "required_for": required_for,
                "required_fields": fields,
                "status": status,
                "active_in_trade_decision": False,
            }
            for req, required_for, fields, status in rows
        ]
    )


def _execution_ledger_requirements() -> pd.DataFrame:
    rows = [
        ("same_day_reference_ledger", "same-day reference replay after formal target stream exists", "missing"),
        ("next_day_fill_ledger", "next-day fill replay with price availability and cost model", "missing"),
        ("cost_turnover_ledger", "transaction cost, turnover, trade days", "missing"),
        ("blocked_fill_ledger", "missing price/cash/no-target fill blockers", "missing"),
    ]
    return pd.DataFrame(
        [{"requirement": req, "detail": detail, "status": status, "formal_model_changed": False} for req, detail, status in rows]
    )


def _reader_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interface": "load_0050_pcf_monthly_anchor(path)",
                "purpose": "Validate and load compact monthly anchor candidate table.",
                "formal_exact": False,
                "trade_decision_changed": False,
            },
            {
                "interface": "resolve_0050_constituents_for_date(date, mode='calendar_month')",
                "purpose": "Resolve the date's yyyy-mm anchor constituent set.",
                "formal_exact": False,
                "trade_decision_changed": False,
            },
            {
                "interface": "resolve_0050_constituents_for_date(date, mode='pit_safe')",
                "purpose": "Resolve latest anchor whose effective_date is not after query date.",
                "formal_exact": False,
                "trade_decision_changed": False,
            },
        ]
    )


def _manifest(anchor: pd.DataFrame, price_req: pd.DataFrame, blockers: pd.DataFrame, monthly_anchor_path: str | Path) -> dict[str, Any]:
    months = int(anchor["effective_month"].nunique())
    anchor_rows = int(len(anchor))
    unique_tickers = int(anchor["ticker"].nunique())
    price_ready = int((price_req["ready_for_backtest_price_only"].astype(str).str.lower() == "true").sum()) if not price_req.empty else 0
    blocking = blockers[blockers["blocks_2014_2023_backtest"].astype(bool)]
    return {
        "task_id": "TASK-BACKTEST-CORE-0050-PIT-CANDIDATE-BACKTEST-DATA-READINESS-20260629",
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "monthly_anchor_path": str(monthly_anchor_path),
        "monthly_anchor_readable": True,
        "covered_months": months,
        "missing_months": max(0, 110 - months),
        "anchor_rows": anchor_rows,
        "unique_anchor_tickers": unique_tickers,
        "price_ready_tickers": price_ready,
        "price_required_tickers": int(len(price_req)),
        "strategy_ready": False,
        "blocking_layer_count": int(len(blocking)),
        "formal_exact": False,
        "source_type": "source_backed_manual_candidate",
        "proxy_rows_mixed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
    }


def _adapter_plan() -> str:
    return """# 0050 PIT candidate adapter plan

Core has a reader for the compact monthly anchor table. Future replay/data-preparation code should use the adapter rather than reading Radar raw PCF daily files directly.

## Modes

- `calendar_month`: returns the constituent set for the query date's yyyy-mm.
- `pit_safe`: returns the latest anchor whose `effective_date <= query_date`.

`calendar_month` is useful for monthly data preparation and coverage checks. Daily replay should prefer `pit_safe` or explicitly audit `anchor_after_query_date` to avoid future-date leakage inside a month.

## Boundaries

This adapter exposes source-backed manual candidate data only. It does not turn PCF/Daily rows into official exact PIT data, and it does not change formal selector or trade decisions.
"""


def _next_experiments_task() -> str:
    return """# Next Experiments/Core Task

Suggested task id:
`TASK-BACKTEST-EXPERIMENTS-2014-2023-PIT-CANDIDATE-DATA-PREP-SMOKE-001`

Scope:
- Load Core monthly anchor candidate through `pcf_pit_candidate_adapter`.
- Verify date resolution in `calendar_month` and `pit_safe` modes.
- Cross-check price coverage for every ticker appearing in monthly anchors.
- Do not run formal strategy replay until Pool1/Pool2 target/signal stream is reconstructed.

Expected conclusion:
- PIT candidate membership layer ready / partial.
- Price gaps listed.
- Signal stream and execution ledger still blockers unless separately completed.
"""


def _summary_zh(manifest: dict[str, Any], blockers: pd.DataFrame) -> str:
    blocking = blockers[blockers["blocks_2014_2023_backtest"].astype(bool)]
    blocker_lines = "\n".join(f"- {row.blocker}: {row.status}" for row in blocking.itertuples())
    return f"""# 0050 PIT candidate backtest data readiness

Core 已建立 0050 monthly anchor PIT candidate reader / readiness layer。

## 結果

- monthly anchor readable：{manifest['monthly_anchor_readable']}
- covered months：{manifest['covered_months']}/110
- anchor rows：{manifest['anchor_rows']}
- unique anchor tickers：{manifest['unique_anchor_tickers']}
- formal_exact=false
- source_type=source_backed_manual_candidate
- formal_model_changed=false
- trade_decision_changed=false

## 是否可供後續回測讀取

可以。後續程式可用 `load_0050_pcf_monthly_anchor` 與 `resolve_0050_constituents_for_date` 讀取指定日期對應的 0050 candidate constituent set。

但目前仍不是完整正式回測 ready。

## 剩餘 blockers

{blocker_lines}
"""


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_bool_false(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n", ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 0050 PCF monthly anchor PIT candidate backtest data readiness outputs.")
    parser.add_argument("--monthly-anchor-path", default=DEFAULT_MONTHLY_ANCHOR_PATH)
    parser.add_argument("--price-coverage-path", default=DEFAULT_PRICE_COVERAGE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_0050_pit_candidate_backtest_data_readiness(
        monthly_anchor_path=args.monthly_anchor_path,
        price_coverage_path=args.price_coverage_path,
        output_dir=args.output_dir,
    )
    print(f"0050_PIT_CANDIDATE_READINESS_OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
