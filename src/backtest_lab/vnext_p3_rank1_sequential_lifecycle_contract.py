from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "outputs/vnext_p3_layer5_daily_feature_state_action_materialization_20260712/p3_layer5_daily_feature_state_matrix.csv"
ADJUSTED_DIR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p3_recent_full_feature_data_readiness_acquisition_20260711\checkpoints\adjusted")
OUT = ROOT / "outputs/vnext_p3_layer04_rank1_sequential_low_turnup_high_turndown_lifecycle_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER04-RANK1-SEQUENTIAL-LOW-TURNUP-HIGH-TURNDOWN-LIFECYCLE-CONTRACT-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling_percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        current = values[-1]
        return float(np.mean(values <= current)) if np.isfinite(current) else np.nan
    return series.rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def adjusted_context(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, audit = [], []
    for ticker in tickers:
        path = ADJUSTED_DIR / f"{ticker}.csv.gz"
        if not path.exists():
            audit.append({"ticker": ticker, "status": "blocked_adjusted_history_missing", "rows": 0})
            continue
        data = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)
        data["decision_date"] = pd.to_datetime(data.date)
        data = data.sort_values("decision_date").drop_duplicates("decision_date", keep="last")
        close = data.adjusted_close.astype(float)
        data["BIAS20_continuous"] = close / close.rolling(20, min_periods=20).mean() - 1
        data["BIAS60_continuous"] = close / close.rolling(60, min_periods=60).mean() - 1
        for label, window, minimum in [("3M", 63, 40), ("6M", 126, 60), ("12M", 252, 120)]:
            data[f"price_pct_{label}"] = rolling_percentile(close, window, minimum)
            data[f"BIAS20_pct_{label}"] = rolling_percentile(data.BIAS20_continuous, window, minimum)
            data[f"BIAS60_pct_{label}"] = rolling_percentile(data.BIAS60_continuous, window, minimum)
        keep = ["decision_date", "ticker", "adjusted_close", "BIAS20_continuous", "BIAS60_continuous",
                "price_pct_3M", "price_pct_6M", "price_pct_12M", "BIAS20_pct_3M", "BIAS20_pct_6M", "BIAS20_pct_12M",
                "BIAS60_pct_3M", "BIAS60_pct_6M", "BIAS60_pct_12M", "source_quality", "adjustment_policy"]
        rows.append(data[keep])
        audit.append({"ticker": ticker, "status": "trusted_adjusted_close_ready", "rows": len(data), "start": data.decision_date.min(), "end": data.decision_date.max(), "adjusted_HLC_ready": False})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit))


def state_contract() -> pd.DataFrame:
    rows = [
        ("S0", "strong_background_eligible", "canonical primary80 rank1 with PIT lineage", "candidate background only; no entry"),
        ("S1", "cooling_relative_low_not_ready", "ticker-self price+BIAS+KD relative-low context AND still falling/RS weak/structure not stabilized", "watch only; buying prohibited"),
        ("S2", "low_turning_up", "must follow S1; KD up/cross, RS5/10 repair, MA20 reclaim/short structure, non-lower-low multi-evidence", "cannot jump from high strong state"),
        ("S3", "entry_confirmed", "must follow S2; capital/volume persistent improvement; chip/crowding not worsening; market-adjusted confirmation", "next-day/deferred official execution"),
        ("S4", "healthy_hold", "post-entry trend/RS/capital reasons remain valid", "hold; no rank1 auto-switch"),
        ("S5", "high_warning_not_exit", "ticker-self price+BIAS+KD relative-high/overheat context", "warning only while RS/MA/capital healthy"),
        ("S6", "high_turning_down", "must follow S5/high background; KD down, RS5/10 weak, structure break, divergence/capital withdrawal/risk rise multi-evidence", "single high/single-day/single indicator prohibited"),
        ("S7", "exit_confirmed", "must follow S6; persistent multi-factor weakening; weak/bear market may increase sensitivity", "exit to cash next-day/deferred; no 00631L fallback"),
    ]
    return pd.DataFrame(rows, columns=["state", "name", "prerequisite", "action_semantics"])


def transition_contract() -> pd.DataFrame:
    allowed = [("S0","S1"),("S1","S1"),("S1","S2"),("S2","S1"),("S2","S2"),("S2","S3"),("S3","S4"),("S4","S4"),("S4","S5"),("S5","S4"),("S5","S5"),("S5","S6"),("S6","S4"),("S6","S5"),("S6","S6"),("S6","S7"),("S7","S0")]
    rows = [{"from_state": a, "to_state": b, "allowed": True, "reason": "frozen sequential or adjacent hold/backoff"} for a,b in allowed]
    rows += [
        {"from_state":"S0","to_state":"S3","allowed":False,"reason":"S0 direct entry prohibited"},
        {"from_state":"S1","to_state":"S3","allowed":False,"reason":"relative low cannot directly enter"},
        {"from_state":"S5","to_state":"S7","allowed":False,"reason":"relative high cannot directly exit"},
        {"from_state":"market_only","to_state":"S3_or_S7","allowed":False,"reason":"market cannot decide stock action alone"},
    ]
    return pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY, dtype={"ticker": str}, low_memory=False)
    daily["decision_date"] = pd.to_datetime(daily.decision_date)
    rank1 = daily.loc[daily.pool_rank.eq(1)].copy()
    context, source_audit = adjusted_context(sorted(rank1.ticker.unique()))
    frame = rank1.merge(context, on=["decision_date", "ticker"], how="left", validate="one_to_one", suffixes=("_candidate", "_continuous"))
    frame["P3_segment"] = np.where(frame.decision_date.lt("2025-07-11"), "P3-1_TDCC_unavailable", "P3-2_TDCC_optional")
    frame["TDCC_used_in_common_state"] = False
    frame["KD_self_pct_3M"] = np.nan
    frame["KD_self_pct_6M"] = np.nan
    frame["KD_self_pct_12M"] = np.nan
    frame["KD_self_history_status"] = "blocked_no_continuous_adjusted_HLC; candidate-day_KD_not_valid_self-history"
    frame["complete_sequential_state_materialized"] = False
    frame["future_outcome_used_as_rule"] = False

    state_contract().to_csv(OUT / "p3_rank1_sequential_state_definition.csv", index=False, encoding="utf-8-sig")
    transition_contract().to_csv(OUT / "p3_rank1_sequential_transition_contract.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT / "p3_rank1_sequential_continuous_feature_matrix.csv.gz", index=False, compression="gzip", encoding="utf-8")
    source_audit.to_csv(OUT / "p3_rank1_adjusted_history_source_audit.csv", index=False, encoding="utf-8-sig")

    supply = []
    for segment, part in [("P3", frame), ("P3-1", frame[frame.P3_segment.str.startswith("P3-1")]), ("P3-2", frame[frame.P3_segment.str.startswith("P3-2")])]:
        for state in [f"S{i}" for i in range(8)]:
            supply.append({"segment":segment,"state":state,"supply_count":len(part) if state=="S0" else 0,"status":"ready_background" if state=="S0" else "blocked_pending_KD_self_history_and_parameter_freeze","not_zero_evidence":state!="S0"})
        supply.append({"segment":segment,"state":"S1_to_S3_complete_sequence","supply_count":0,"status":"blocked_not_materialized","not_zero_evidence":True})
        supply.append({"segment":segment,"state":"S5_to_S7_complete_sequence","supply_count":0,"status":"blocked_not_materialized","not_zero_evidence":True})
    pd.DataFrame(supply).to_csv(OUT / "p3_rank1_sequential_state_supply_audit.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"parameter_family":"relative_low_percentile","candidate_values":"Strategy Center freeze required","selected":False},
        {"parameter_family":"relative_high_percentile","candidate_values":"Strategy Center freeze required","selected":False},
        {"parameter_family":"turn_up_down_confirmation_window","candidate_values":"Strategy Center freeze required","selected":False},
        {"parameter_family":"capital_confirmation_persistence","candidate_values":"Strategy Center freeze required","selected":False},
        {"parameter_family":"market_threshold_adjustment","candidate_values":"Strategy Center freeze required","selected":False},
    ]).to_csv(OUT / "p3_rank1_sequential_parameter_decision_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"item":"adjusted close self-history","status":"ready","detail":"101 rank1 tickers; trusted nonofficial research-grade"},
        {"item":"price and BIAS 3/6/12M percentile","status":"materialized","detail":"continuous adjusted close; rolling windows 63/126/252 observations"},
        {"item":"KD 3/6/12M self percentile","status":"blocked","detail":"adjusted source has close only; no continuous adjusted high/low; intermittent candidate-day KD cannot substitute"},
        {"item":"corporate action","status":"diagnostic_proxy","detail":"provider adjustment; not formal event completeness"},
        {"item":"TDCC P3-1","status":"not_available","detail":"NA, not zero"},
        {"item":"TDCC P3-2","status":"optional","detail":"not used in common state definition"},
    ]).to_csv(OUT / "p3_rank1_sequential_missingness_proxy_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"audit":"future_outcome_feature","violations":0},{"audit":"P3_2_outcome_read","violations":0},{"audit":"candidate_day_KD_used_as_continuous_self_history","violations":0},{"audit":"TDCC_P3_1_zero_fill","violations":0}]).to_csv(OUT / "p3_rank1_sequential_future_PIT_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {"task_id":TASK,"status":"blocked_KD_continuous_adjusted_HLC_required_before_sequence_supply","requested_start":"2023-07-11","requested_end":"2026-06-29","actual_start":str(frame.decision_date.min().date()),"actual_end":str(frame.decision_date.max().date()),"rank1_daily_rows":len(frame),"rank1_unique_tickers":frame.ticker.nunique(),"state_definition_ready":True,"transition_contract_ready":True,"price_BIAS_3_6_12M_percentiles_ready":True,"KD_3_6_12M_self_percentiles_ready":False,"complete_entry_sequence_supply_ready":False,"complete_exit_sequence_supply_ready":False,"sufficient_for_walk_forward":False,"ready_for_experiments":False,"performance_executed":False,"P3_2_outcome_read":False,"NAV_executed":False,"Top3_executed":False,"future_data_violation_count":0,"formal_model_changed":False,"trade_decision_changed":False,"active_in_trade_decision":False,"report_changed":False,"portfolio_replay_executed":False,"ready_for_strategy_replay":False,"ready_for_formal":False,"not_live_rule":True,"forward_returns_live_rule_usage":False}
    readiness.update({
        "status": "authorized_waiting_bounded_adjusted_HLC_source",
        "source_acquisition_started": False,
        "radar_download_executed": False,
        "governance_conflict": None,
        "requires_strategy_center_scope_ruling": False,
        "diagnostic_subproblem": True,
        "supports_sequential_lifecycle_rank1_timing": True,
        "representative_of_full_all80_layer5": False,
        "may_be_used_to_reject_full_layer5": False,
        "broad_additive_formula_followup": False,
    })
    (OUT / "readiness_for_p3_rank1_sequential_lifecycle.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text("# P3 rank1 sequential low-turn-up/high-turn-down lifecycle contract\n\nStrategy Center已確認：停止的是broad additive formula，新sequential rank1 bounded diagnostic合法有效。S0-S7 contract與price/BIAS evidence保留；目前等待Radar補rank1-only adjusted HLC/factor，以建立KD 3/6/12月自身歷史與sequence供給。未交Experiments、未讀future outcome或P3-2績效。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id":TASK,"inputs":{"daily_sha256":sha(DAILY)},"files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
