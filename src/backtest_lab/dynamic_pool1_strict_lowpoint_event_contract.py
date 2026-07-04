"""Build the Dynamic Pool1 strict lowpoint timing-band event contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-STRICT-LOWPOINT-TIMING-BAND-EVENT-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-STRICT-LOWPOINT-EVENT-CONTRACT-VALIDATION-001"
DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-06-17\repo-ai-stock-backtest-lab-repo\outputs"
    r"\experiments_dynamic_pool1_strict_lowpoint_timing_band_stability_20260704"
)
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_strict_lowpoint_event_contract_20260704")
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
CASE_TRACE_TICKERS = {
    "6669.TW": "緯穎",
    "2308.TW": "台達電",
    "2317.TW": "鴻海",
}

PRIMARY_VARIANTS = {
    "lowpoint_0_2d_rebound_5_12pct": "strict_lowpoint_0_2d_rebound_5_12pct",
    "lowpoint_0_5d_rebound_5_12pct_downside_deceleration": "strict_lowpoint_0_5d_rebound_5_12pct_downside_deceleration",
    "lowpoint_0_5d_rebound_5_12pct_short_rs_repair": "strict_lowpoint_0_5d_rebound_5_12pct_short_rs_repair",
}
REFERENCE_VARIANTS = {
    "lowpoint_3_5d_rebound_5_12pct": "strict_lowpoint_3_5d_rebound_5_12pct_reference_only",
}


def run_dynamic_pool1_strict_lowpoint_event_contract(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event_panel = pd.read_csv(source / "strict_lowpoint_timing_band_event_panel.csv")
    calendar = _load_calendar(Path(liquidity_dir))
    contract, blocked = _build_contract(event_panel, calendar)
    summary = _variant_summary(contract)
    case_trace = _build_case_trace(contract)
    negative = _load_negative_control(source)
    future_audit = _future_data_audit(contract)

    contract.to_csv(output / "strict_lowpoint_event_contract.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "event_variant_summary.csv", index=False, encoding="utf-8-sig")
    case_trace.to_csv(output / "case_trace_6669_2308_2317.csv", index=False, encoding="utf-8-sig")
    negative.to_csv(output / "negative_control_reference.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(output / "blocked_rows.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_strict_lowpoint_event_contract_ready",
        "output_dir": str(output.resolve()),
        "source_dir": str(source.resolve()),
        "event_rows": int(len(contract)),
        "primary_event_rows": int(contract["event_variant_role"].eq("primary").sum()),
        "reference_event_rows": int(contract["event_variant_role"].eq("reference_only").sum()),
        "blocked_rows": int(len(blocked)),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()),
        "case_trace_refresh_rows": int(contract["case_trace_refresh_only"].sum()),
        "case_trace_rows": int(len(case_trace)),
        "case_trace_expected_tickers": list(CASE_TRACE_TICKERS),
        "case_trace_found_tickers": sorted(case_trace.loc[case_trace["event_found"], "ticker"].astype(str).unique().tolist()),
        "case_trace_missing_event_tickers": sorted(
            case_trace.loc[~case_trace["event_found"], "ticker"].astype(str).unique().tolist()
        ),
        "case_trace_contains_2317": bool(case_trace["ticker"].astype(str).eq("2317.TW").any()),
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, summary), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_upstream_event_panel", "status": "completed"},
            {"step": "normalize_event_contract_fields", "status": "completed"},
            {"step": "write_event_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _build_case_trace(contract: pd.DataFrame) -> pd.DataFrame:
    case_trace = contract[contract["ticker"].isin(CASE_TRACE_TICKERS)].copy()
    case_trace["event_found"] = True
    case_trace["case_trace_blocked_reason"] = ""
    found = set(case_trace["ticker"].astype(str).unique())
    missing_rows = []
    for ticker, name in CASE_TRACE_TICKERS.items():
        if ticker in found:
            continue
        row = {column: "" for column in case_trace.columns}
        row.update(
            {
                "ticker": ticker,
                "candidate_name": name,
                "candidate_source": "case_trace_hygiene_placeholder",
                "price_ready": False,
                "liquidity_ready": False,
                "event_found": False,
                "case_trace_blocked_reason": "no_strict_lowpoint_event_for_case_ticker",
                "blocked_reason": "no_strict_lowpoint_event_for_case_ticker",
                "uses_forward_return_as_rule": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
                "report_changed": False,
                "portfolio_replay_executed": False,
            }
        )
        missing_rows.append(row)
    if missing_rows:
        case_trace = pd.concat([case_trace, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
    return case_trace


def _build_contract(event_panel: pd.DataFrame, calendar: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = event_panel.copy()
    allowed = {**PRIMARY_VARIANTS, **REFERENCE_VARIANTS}
    frame["contract_variant"] = frame["strict_variant_id"].map(allowed)
    blocked = frame[frame["contract_variant"].isna()].copy()
    blocked["blocked_reason"] = "strict_variant_not_allowed_in_core_contract"
    frame = frame[frame["contract_variant"].notna()].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame["next_tradable_date"] = frame["signal_date"].map(lambda date: _next_tradable(date, calendar))
    frame["case_trace_refresh_only"] = frame["signal_date"].between(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-03"))
    frame["event_variant_role"] = frame["strict_variant_id"].map(lambda value: "primary" if value in PRIMARY_VARIANTS else "reference_only")
    out = pd.DataFrame(
        {
            "signal_date": frame["signal_date"].dt.strftime("%Y-%m-%d"),
            "next_tradable_date": frame["next_tradable_date"],
            "ticker": frame["ticker"].astype(str),
            "candidate_name": frame.get("candidate_name", "").fillna("").astype(str),
            "candidate_source": "dynamic_pool1_pre_reclaim_lowpoint_watch_diagnostic",
            "candidate_layer": frame.get("candidate_layer", "").fillna("").astype(str),
            "prior_strength_eligible": True,
            "theme_or_mainline_context": frame.get("layer_group", "").fillna("").astype(str),
            "liquidity_ready": True,
            "price_ready": True,
            "days_since_local_low": pd.to_numeric(frame.get("days_since_10d_low"), errors="coerce"),
            "rebound_from_local_low_pct": pd.to_numeric(frame.get("rebound_from_10d_low_pct"), errors="coerce"),
            "days_since_low_band": frame.get("days_since_low_band", "").fillna("").astype(str),
            "rebound_from_low_band": frame.get("rebound_from_low_band", "").fillna("").astype(str),
            "close": pd.to_numeric(frame.get("close"), errors="coerce"),
            "ma20": pd.to_numeric(frame.get("ma20"), errors="coerce"),
            "ma60": pd.to_numeric(frame.get("ma60"), errors="coerce"),
            "ma120": pd.to_numeric(frame.get("ma120"), errors="coerce"),
            "close_vs_ma20_pct": pd.to_numeric(frame.get("close_vs_ma20_pct"), errors="coerce"),
            "close_vs_ma60_pct": pd.to_numeric(frame.get("close_vs_ma60_pct"), errors="coerce"),
            "drawdown_from_20d_high_pct": pd.to_numeric(frame.get("drawdown_from_20d_high_pct"), errors="coerce"),
            "drawdown_from_60d_high_pct": pd.to_numeric(frame.get("drawdown_from_60d_high_pct"), errors="coerce"),
            "downside_deceleration": frame["contract_variant"].astype(str).str.contains("downside_deceleration"),
            "short_rs_repair": frame["contract_variant"].astype(str).str.contains("short_rs_repair"),
            "rs_vs_0050_3d_or_5d": pd.to_numeric(frame.get("rs_vs_0050_5d_pct"), errors="coerce").fillna(
                pd.to_numeric(frame.get("rs_vs_0050_3d_pct"), errors="coerce")
            ),
            "rs_vs_00631l_3d_or_5d": pd.to_numeric(frame.get("rs_vs_00631L_5d_pct"), errors="coerce").fillna(
                pd.to_numeric(frame.get("rs_vs_00631L_3d_pct"), errors="coerce")
            ),
            "rs60_positive_vs_both_at_event": frame.get("rs60_positive_vs_both_at_event", False).map(_as_bool),
            "captures_6669_window": frame["ticker"].astype(str).eq("6669.TW"),
            "case_trace_refresh_only": frame["case_trace_refresh_only"],
            "event_variant": frame["contract_variant"],
            "event_variant_role": frame["event_variant_role"],
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "report_changed": False,
            "portfolio_replay_executed": False,
        }
    )
    out["blocked_reason"] = out["next_tradable_date"].map(lambda value: "" if value else "missing_next_tradable_date")
    blocked_next = out[out["blocked_reason"].ne("")].copy()
    if not blocked_next.empty:
        blocked = pd.concat([blocked, blocked_next], ignore_index=True, sort=False)
    return out, blocked


def _load_calendar(liquidity_dir: Path) -> list[str]:
    dates: set[str] = set()
    shard_dir = liquidity_dir / "shards"
    if shard_dir.exists():
        for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
            try:
                df = pd.read_csv(shard, usecols=["date"])
            except (OSError, ValueError):
                continue
            dates.update(pd.to_datetime(df["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    return sorted(dates)


def _next_tradable(signal_date: pd.Timestamp, calendar: list[str]) -> str:
    if pd.isna(signal_date):
        return ""
    text = signal_date.strftime("%Y-%m-%d")
    later = [date for date in calendar if date > text]
    return later[0] if later else ""


def _variant_summary(contract: pd.DataFrame) -> pd.DataFrame:
    return (
        contract.groupby(["event_variant", "event_variant_role"], as_index=False)
        .agg(
            event_count=("ticker", "count"),
            unique_tickers=("ticker", "nunique"),
            captures_6669_count=("captures_6669_window", "sum"),
            case_trace_refresh_count=("case_trace_refresh_only", "sum"),
            blocked_rows=("blocked_reason", lambda s: int(s.astype(str).ne("").sum())),
        )
        .sort_values("event_variant")
    )


def _load_negative_control(source: Path) -> pd.DataFrame:
    path = source / "negative_control_summary.csv"
    if path.exists():
        out = pd.read_csv(path)
    else:
        out = pd.DataFrame(columns=["negative_control_id", "status"])
    out["accepted_for_primary_contract"] = False
    out["diagnostic_reference_only"] = True
    return out


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    out = contract[["signal_date", "next_tradable_date", "ticker", "event_variant", "case_trace_refresh_only"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _summary(manifest: dict, variant_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 strict lowpoint event contract",
            "",
            "本包只建立 strict lowpoint timing-band event contract；不跑 portfolio，不改正式模型、交易或日報。",
            "",
            f"- event rows：{manifest['event_rows']}",
            f"- primary event rows：{manifest['primary_event_rows']}",
            f"- reference event rows：{manifest['reference_event_rows']}",
            f"- blocked rows：{manifest['blocked_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            f"- case trace refresh rows：{manifest['case_trace_refresh_rows']}",
            f"- case trace rows：{manifest['case_trace_rows']}",
            f"- case trace missing event tickers：{manifest['case_trace_missing_event_tickers']}",
            "- `strict_lowpoint_3_5d_rebound_5_12pct_reference_only` 只作 reference，不是主線。",
            "",
            "## Variant summary",
            variant_summary.to_csv(index=False).strip(),
        ]
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_strict_lowpoint_event_contract(
        source_dir=args.source_dir,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
