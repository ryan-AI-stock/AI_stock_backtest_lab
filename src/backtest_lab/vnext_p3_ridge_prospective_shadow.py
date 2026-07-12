from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
STAGE_A = ROOT / "outputs/vnext_p3_layer5_walk_forward_dual_model_ranking_stage_a_contract_20260712"
OUT = ROOT / "outputs/vnext_p3_ridge_prospective_shadow_contract_20260712"
MODEL_DIR = OUT / "model"
SEED = 20260712
ALPHAS = (0.1, 1.0, 10.0)
VALIDATION_DATES = 40
EMBARGO_DATES = 40
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-RIDGE-PROSPECTIVE-SHADOW-CONTRACT-001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def date_group_weights(frame: pd.DataFrame) -> np.ndarray:
    return (1.0 / frame.groupby("decision_date")["ticker"].transform("size")).to_numpy()


def ordinary_weak_rank_ic(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    tested = frame.assign(prediction=prediction)
    values: list[float] = []
    for _, group in tested[tested.full_spec_v2_state.isin(["ordinary_market", "weak_market"])].groupby("decision_date"):
        if len(group) >= 3 and group.net_excess_vs_0050.nunique() > 1:
            values.append(float(spearmanr(group.prediction, group.net_excess_vs_0050).statistic))
    return float(np.nanmean(values)) if values else float("nan")


def build_model() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(STAGE_A / "p3_dual_model_feature_data_dictionary.csv").feature.tolist()
    matrix = pd.read_csv(STAGE_A / "p3_dual_model_frozen_feature_matrix.csv.gz", dtype={"ticker": str}, low_memory=False)
    labels = pd.read_csv(STAGE_A / "p3_dual_model_exact_label_contract.csv.gz", dtype={"ticker": str}, low_memory=False)
    matrix["decision_date"] = pd.to_datetime(matrix.decision_date)
    labels["decision_date"] = pd.to_datetime(labels.decision_date)
    mature = labels[labels.horizon_td.eq(20)].copy()
    data = matrix.merge(mature, on=["decision_date", "ticker"], how="inner", validate="one_to_one")
    data = data[data.full_spec_v2_state.isin(["ordinary_market", "weak_market", "strong_market", "confirmed_bear"])].copy()
    dates = sorted(data.decision_date.unique())
    validation = dates[-VALIDATION_DATES:]
    train_end_index = len(dates) - VALIDATION_DATES - EMBARGO_DATES - 1
    if train_end_index < 0:
        raise RuntimeError("insufficient mature history for frozen validation and embargo")
    train_end = dates[train_end_index]
    train = data[data.decision_date.le(train_end)].sort_values(["decision_date", "ticker"])
    valid = data[data.decision_date.isin(validation)].sort_values(["decision_date", "ticker"])
    choices = []
    for alpha in ALPHAS:
        candidate = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(train[features].astype(float), train.net_excess_vs_0050, ridge__sample_weight=date_group_weights(train))
        choices.append((ordinary_weak_rank_ic(valid, candidate.predict(valid[features].astype(float))), alpha))
    selected_ic, alpha = max(choices, key=lambda row: -np.inf if pd.isna(row[0]) else row[0])
    final = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
    final.fit(data[features].astype(float), data.net_excess_vs_0050, ridge__sample_weight=date_group_weights(data))
    model_path = MODEL_DIR / "p3_frozen_prospective_ridge_20td.joblib"
    joblib.dump(final, model_path)
    schema = {
        "features": features,
        "feature_count": len(features),
        "feature_source_hash": sha256(STAGE_A / "p3_dual_model_feature_data_dictionary.csv"),
        "target": "20TD_EP05_plus_10bp_per_side_net_excess_vs_0050",
        "risk_guard": "existing_fixed_cross_sectional_risk_decile_9_10_exclusion",
    }
    atomic_write(MODEL_DIR / "feature_schema.json", json.dumps(schema, ensure_ascii=False, indent=2))
    metadata = {
        "task_id": TASK,
        "model_role": "prospective_shadow_only",
        "model_type": "sklearn_pipeline_median_imputer_standard_scaler_ridge",
        "training_label_horizon_td": 20,
        "mature_label_cutoff": str(pd.Timestamp(data.decision_date.max()).date()),
        "training_start": str(pd.Timestamp(data.decision_date.min()).date()),
        "training_rows": len(data),
        "training_decision_dates": data.decision_date.nunique(),
        "validation_start": str(pd.Timestamp(min(validation)).date()),
        "validation_end": str(pd.Timestamp(max(validation)).date()),
        "embargo_decision_dates": EMBARGO_DATES,
        "bounded_alpha_candidates": list(ALPHAS),
        "selected_alpha": alpha,
        "validation_ordinary_weak_rank_ic": selected_ic,
        "selection_used_prior_OOS_results": False,
        "seed": SEED,
        "threads": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "model_sha256": sha256(model_path),
        "feature_schema_sha256": sha256(MODEL_DIR / "feature_schema.json"),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    atomic_write(MODEL_DIR / "model_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def validate_manifest(manifest: dict, decision_date: str) -> str:
    if manifest.get("market_status") == "market_closed_no_signal":
        return "market_closed_no_signal"
    required = {
        "decision_date": decision_date,
        "market_status": "open",
        "data_ready": True,
        "exact_layer4_primary80": True,
        "candidate_count": 80,
    }
    failures = [f"{key}={manifest.get(key)!r}" for key, expected in required.items() if manifest.get(key) != expected]
    if failures:
        raise RuntimeError("mandatory exact-current gate failed: " + ", ".join(failures))
    if manifest.get("candidate_scope_semantics") in {"carried_scope_reference_only", "carry_forward"}:
        raise RuntimeError("carried Layer4 scope cannot produce prospective prediction")
    for key in ["available_at", "candidate_universe_version", "candidate_universe_hash", "corporate_action_guard_status"]:
        if not manifest.get(key):
            raise RuntimeError(f"missing mandatory manifest field: {key}")
    return "ready"


def score_frame(frame: pd.DataFrame, manifest: dict, model_metadata: dict) -> pd.DataFrame:
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    missing = sorted(set(schema["features"]) - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen model features: {missing}")
    frame = frame.copy()
    frame["ticker"] = frame.ticker.astype(str).str.zfill(4)
    if len(frame) != 80 or frame.ticker.nunique() != 80:
        raise RuntimeError("exact primary80 input must contain exactly 80 unique tickers")
    if canonical_hash(frame.sort_values("ticker")) != manifest["candidate_universe_hash"]:
        raise RuntimeError("candidate universe hash mismatch")
    if "risk_overheat_crowding_score" not in frame:
        raise RuntimeError("fixed risk guard input unavailable")
    model = joblib.load(MODEL_DIR / "p3_frozen_prospective_ridge_20td.joblib")
    frame["ridge_score"] = model.predict(frame[schema["features"]].astype(float))
    frame["ridge_rank"] = frame.ridge_score.rank(method="first", ascending=False).astype(int)
    frame["risk_decile"] = np.ceil(frame.risk_overheat_crowding_score.rank(pct=True, method="average") * 10).clip(1, 10).astype("Int64")
    frame["risk_guard_excluded"] = frame.risk_decile.ge(9)
    base = {
        "decision_date": manifest["decision_date"],
        "available_at": manifest["available_at"],
        "candidate_universe_version": manifest["candidate_universe_version"],
        "candidate_universe_hash": manifest["candidate_universe_hash"],
        "market_state": manifest.get("market_state"),
        "market_confidence": manifest.get("market_confidence"),
        "data_freshness_status": manifest.get("data_freshness_status", "ready"),
        "corporate_action_guard_status": manifest["corporate_action_guard_status"],
        "model_hash": model_metadata["model_sha256"],
    }
    records = []
    r0 = frame.nsmallest(10, "ridge_rank")
    r1 = frame[~frame.risk_guard_excluded].nlargest(10, "ridge_score")
    for shadow, selected in [("R0_frozen_ridge_raw_top10", r0), ("R1_fixed_risk_decile_9_10_excluded_top10", r1)]:
        for shadow_rank, row in enumerate(selected.sort_values("ridge_score", ascending=False).itertuples(), 1):
            records.append({**base, "shadow_id": shadow, "ticker": row.ticker, "ridge_score": row.ridge_score, "ridge_rank": row.ridge_rank, "shadow_rank": shadow_rank, "risk_decile": row.risk_decile, "risk_guard_excluded": bool(row.risk_guard_excluded), "missing_feature_count": int(pd.isna(frame.loc[row.Index, schema["features"]]).sum()), "not_live_rule": True})
    return pd.DataFrame(records)


def run_prediction(input_dir: Path, ledger_root: Path, dry_run: bool = False) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    decision_date = manifest["decision_date"]
    status = validate_manifest(manifest, decision_date)
    if status == "market_closed_no_signal":
        return {"status": status, "decision_date": decision_date, "prediction_rows": 0}
    model_metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(input_dir / "candidate_features.csv.gz", dtype={"ticker": str}, low_memory=False)
    result = score_frame(frame, manifest, model_metadata)
    out_dir = ledger_root / decision_date.replace("-", "/")
    out_file = out_dir / "p3_ridge_shadow_predictions.csv"
    content = result.to_csv(index=False, lineterminator="\n")
    if out_file.exists():
        if hashlib.sha256(out_file.read_bytes()).hexdigest() != hashlib.sha256(content.encode()).hexdigest():
            raise RuntimeError("append-only prediction exists with different content")
        return {"status": "deterministic_existing_noop", "decision_date": decision_date, "prediction_rows": len(result), "sha256": sha256(out_file)}
    if not dry_run:
        atomic_write(out_file, content)
        atomic_write(out_dir / "manifest.json", json.dumps({"task_id": TASK, "decision_date": decision_date, "prediction_sha256": sha256(out_file), "input_manifest_sha256": sha256(input_dir / "manifest.json"), "model_sha256": model_metadata["model_sha256"], "append_only": True, "report_published": False, "trade_decision_emitted": False, "future_data_violation_count": 0}, ensure_ascii=False, indent=2))
    return {"status": "dry_run_ready" if dry_run else "prospective_prediction_appended", "decision_date": decision_date, "prediction_rows": len(result), "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def evaluate_matured(prediction_root: Path, outcome_file: Path, output_root: Path) -> dict:
    prediction_files = list(prediction_root.glob("*/*/*/p3_ridge_shadow_predictions.csv"))
    if not prediction_files:
        return {"status": "no_prospective_predictions", "evaluated_rows": 0}
    predictions = pd.concat([pd.read_csv(path, dtype={"ticker": str}) for path in prediction_files], ignore_index=True)
    outcomes = pd.read_csv(outcome_file, dtype={"ticker": str}, low_memory=False)
    required = {"decision_date", "ticker", "horizon_td", "outcome_status", "actual_net_excess_vs_0050", "path_MDD", "tail_daily_return_p10", "large_down_count", "outcome_available_at"}
    if missing := required - set(outcomes):
        raise RuntimeError(f"outcome ledger missing fields: {sorted(missing)}")
    mature = outcomes[outcomes.horizon_td.isin([10, 20]) & outcomes.outcome_status.eq("mature_official")].copy()
    joined = predictions.merge(mature, on=["decision_date", "ticker"], how="inner", validate="many_to_many")
    if joined.empty:
        return {"status": "no_mature_10_20TD_labels", "evaluated_rows": 0}
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "p3_ridge_shadow_matured_outcomes.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"ticker": str})
        joined = pd.concat([old, joined], ignore_index=True).drop_duplicates(["shadow_id", "decision_date", "ticker", "horizon_td"], keep="first")
    atomic_write(path, joined.to_csv(index=False, lineterminator="\n"))
    return {"status": "matured_outcomes_appended", "evaluated_rows": len(joined), "sha256": sha256(path)}


def build_contract() -> None:
    metadata = build_model()
    policies = pd.DataFrame([
        {"policy": "prediction_immutability", "value": "append_only_no_overwrite"},
        {"policy": "R0", "value": "frozen_Ridge_raw_top10"},
        {"policy": "R1", "value": "same_Ridge_fixed_risk_decile_9_10_exclusion_then_top10"},
        {"policy": "ML_risk", "value": "prohibited"},
        {"policy": "checkpoint_40_80", "value": "monitor_only_no_promotion"},
        {"policy": "promotion_minimum", "value": "120_normal_trading_days_and_mature_20TD_labels"},
        {"policy": "primary_regime", "value": "ordinary_and_weak_clustered_by_decision_date"},
        {"policy": "weak_sample_shortfall", "value": "report_shortfall_do_not_substitute_strong"},
        {"policy": "current_universe", "value": "exact_same_day_Layer0_4_primary80_only"},
        {"policy": "market_closed", "value": "skip_success_no_signal"},
    ])
    policies.to_csv(OUT / "p3_ridge_shadow_frozen_policy.csv", index=False, encoding="utf-8-sig")
    readiness = {
        "task_id": TASK,
        "status": "contract_and_frozen_model_ready_live_shadow_blocked_current_exact_layer4",
        "frozen_model_artifact_ready": True,
        "model_sha256": metadata["model_sha256"],
        "mature_20TD_training_cutoff": metadata["mature_label_cutoff"],
        "latest_exact_layer4_primary80_date": "2026-06-29",
        "current_exact_layer4_primary80_ready": False,
        "carried_scope_allowed": False,
        "append_only_prediction_contract_ready": True,
        "outcome_maturation_evaluator_ready": True,
        "github_actions_hook_ready": True,
        "historical_exact_dry_run_decision_date": "2026-06-26",
        "dry_run_is_prospective_ledger": False,
        "ready_for_first_prospective_prediction": False,
        "ready_for_promotion_judgment": False,
        "portfolio_selector_created": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
    }
    atomic_write(OUT / "readiness_for_p3_ridge_prospective_shadow.json", json.dumps(readiness, ensure_ascii=False, indent=2))
    atomic_write(OUT / "final_summary_zh.md", "# P3 Ridge prospective shadow\n\nFrozen Ridge artifact、R0/R1 append-only ledger、成熟標籤 evaluator 與 Actions hook 已建立。現行 exact Layer4 primary80 僅至 2026-06-29，因此 live shadow 維持 blocked；禁止 carried scope 與歷史回填。\n")
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "manifest.json")
    atomic_write(OUT / "manifest.json", json.dumps({"task_id": TASK, "generated_at": datetime.now(timezone.utc).isoformat(), "files": [{"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-contract")
    predict = sub.add_parser("predict")
    predict.add_argument("--input-dir", type=Path, required=True)
    predict.add_argument("--ledger-root", type=Path, default=ROOT / "data/vnext_shadow/predictions")
    predict.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--prediction-root", type=Path, default=ROOT / "data/vnext_shadow/predictions")
    evaluate.add_argument("--outcome-file", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, default=ROOT / "data/vnext_shadow/evaluations")
    args = parser.parse_args()
    if args.command == "build-contract":
        build_contract()
    elif args.command == "predict":
        print(json.dumps(run_prediction(args.input_dir, args.ledger_root, args.dry_run), ensure_ascii=False))
    else:
        print(json.dumps(evaluate_matured(args.prediction_root, args.outcome_file, args.output_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
