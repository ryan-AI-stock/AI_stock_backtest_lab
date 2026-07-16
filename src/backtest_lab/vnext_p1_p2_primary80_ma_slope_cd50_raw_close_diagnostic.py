from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_action_legs import (
    SHIFTED_RAW_BLOCKED,
    active_candidate_panel,
    feature_panel,
    load_prices,
    ranked_candidates,
    simulate_actions,
)
from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_contract import TASK, parameter_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p1_p2_primary80_MA_slope_CD50_official_raw_close_diagnostic_20260716"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_close_features(raw: pd.DataFrame) -> pd.DataFrame:
    analysis = raw.copy()
    analysis["source_quality"] = analysis.source_quality.astype(str) + ";official_raw_close_intentional_diagnostic"
    features = feature_panel(analysis)
    features["raw_close_return"] = features.groupby(["period", "ticker"], sort=False).value.pct_change(fill_method=None)
    features["corporate_action_or_scale_discontinuity"] = features.raw_close_return.abs().gt(0.15)
    features["corporate_action_guard_60obs"] = (
        features.groupby(["period", "ticker"], sort=False).corporate_action_or_scale_discontinuity
        .transform(lambda s: s.rolling(60, min_periods=1).max().astype(bool))
    )
    features["history_ready"] &= ~features.corporate_action_guard_60obs
    return features


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("materialize_raw_close_diagnostic\n", encoding="utf-8")
    _, raw = load_prices()
    features = raw_close_features(raw)
    panel = active_candidate_panel(features)
    candidates = ranked_candidates(panel)
    actions, requirements, blocked = simulate_actions(features, candidates, raw)

    guard = features.loc[features.corporate_action_guard_60obs, ["period", "ticker", "date", "raw_close_return", "corporate_action_or_scale_discontinuity"]]
    held = actions.loc[actions.incumbent.notna(), ["variant_id", "period", "decision_date", "incumbent"]].rename(
        columns={"decision_date": "date", "incumbent": "ticker"}
    )
    held_guard = held.merge(guard, on=["period", "ticker", "date"], how="inner").drop_duplicates()
    no_observation = actions.loc[actions.action.eq("hold_no_ticker_observation")].copy()

    shifted_blocked = pd.read_csv(SHIFTED_RAW_BLOCKED, dtype={"ticker": str})
    shifted_blocked["ticker"] = shifted_blocked.ticker.str.zfill(4)
    shifted_blocked["requested_execution_date"] = pd.to_datetime(shifted_blocked.date)
    source_class = shifted_blocked[["period", "ticker", "requested_execution_date", "blocked_reason_after_bounded_network"]].drop_duplicates()
    if len(blocked):
        blocked = blocked.merge(source_class, on=["period", "ticker", "requested_execution_date"], how="left")
        blocked["precise_source_class"] = blocked.blocked_reason_after_bounded_network.fillna(
            "exact_raw_close_absent_after_local_only_close_basis_rechain"
        )
    else:
        blocked["precise_source_class"] = pd.Series(dtype=str)

    rows = []
    for variant in parameter_matrix().variant_id:
        b = blocked.loc[blocked.variant_id.eq(variant)]
        g = held_guard.loc[held_guard.variant_id.eq(variant)]
        n = no_observation.loc[no_observation.variant_id.eq(variant)]
        req = requirements.loc[requirements.variant_id.eq(variant)]
        ready = len(b) == 0 and len(g) == 0 and len(n) == 0
        rows.append(
            {
                "variant_id": variant,
                "execution_legs_ready": len(req),
                "execution_blocked_rows": len(b),
                "corporate_action_guard_held_rows": len(g),
                "incumbent_no_close_observation_rows": len(n),
                "exact_path_coverage_pass": ready,
                "ready_for_experiments": ready,
            }
        )
    readiness_table = pd.DataFrame(rows)
    ready_variants = readiness_table.loc[readiness_table.ready_for_experiments, "variant_id"].tolist()

    features.to_csv(OUT / "raw_close_MA_slope_feature_guard_compact.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "raw_close_candidate_rank_compact.csv.gz", index=False, compression="gzip")
    actions.to_csv(OUT / "raw_close_CD50_action_trace.csv.gz", index=False, compression="gzip")
    requirements.to_csv(OUT / "raw_close_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUT / "raw_close_execution_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    held_guard.to_csv(OUT / "raw_close_corporate_action_guard_held_path_ledger.csv.gz", index=False, compression="gzip")
    no_observation.to_csv(OUT / "raw_close_incumbent_no_observation_ledger.csv.gz", index=False, compression="gzip")
    readiness_table.to_csv(OUT / "raw_close_CD50_per_variant_readiness.csv", index=False, encoding="utf-8-sig")
    policy = {
        "analysis_price_basis": "official_raw_close_intentional_diagnostic",
        "raw_as_adjusted_fallback": False,
        "total_return_basis": False,
        "formal_basis": False,
        "non_close_family_used": False,
        "new_radar_download_authorized": False,
        "corporate_action_guard": "abs_same_ticker_raw_close_return_gt_15pct_blocks_current_and_following_59_observations",
        "neighbor_price_substitution": False,
        "fixed_variants": 50,
    }
    (OUT / "raw_close_diagnostic_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {
        "task_id": TASK,
        "status": "ready_for_experiments" if ready_variants else "blocked_no_variant_has_complete_raw_close_and_corporate_action_safe_path",
        "analysis_price_basis": "official_raw_close_intentional_diagnostic",
        "fixed_variants": 50,
        "ready_variant_count": len(ready_variants),
        "ready_variants": ready_variants,
        "execution_blocked_rows": len(blocked),
        "execution_blocked_unique_keys": len(blocked.drop_duplicates(["period", "ticker", "role", "requested_execution_date"])) if len(blocked) else 0,
        "corporate_action_guard_held_rows": len(held_guard),
        "incumbent_no_close_observation_rows": len(no_observation),
        "ready_for_experiments": bool(ready_variants),
        "further_radar_probe_authorized": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_raw_close_diagnostic.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        "# MA-slope CD50 official raw-close diagnostic\n\n"
        f"- ready variants: {len(ready_variants)}/50\n"
        f"- execution blocked rows: {len(blocked)}\n"
        f"- corporate-action guarded held rows: {len(held_guard)}\n"
        f"- incumbent no-close rows: {len(no_observation)}\n"
        "- intentional raw-close diagnostic; not adjusted, total-return, or formal.\n"
        "- no further Radar download authorized.\n",
        encoding="utf-8",
    )
    (OUT / "current_step.txt").write_text("ready_for_experiments_handoff\n" if ready_variants else "blocked_no_further_download\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(
        json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
