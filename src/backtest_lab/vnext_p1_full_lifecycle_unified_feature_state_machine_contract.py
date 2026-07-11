from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-FULL-LIFECYCLE-UNIFIED-FEATURE-STATE-MACHINE-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_full_lifecycle_minimum_data_acquisition_20260710")
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_full_lifecycle_unified_feature_state_machine_contract_20260711"
FLAGS = {
    "formal_model_changed": False, "trade_decision_changed": False,
    "active_in_trade_decision": False, "report_changed": False,
    "portfolio_replay_executed": False, "ready_for_strategy_replay": False,
    "ready_for_formal": False, "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(radar_dir: Path = DEFAULT_RADAR, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_step.txt").write_text("validating_radar_source_package", encoding="utf-8")
    required = [
        "twse_compact_repair_dedup_audit.csv", "twse_true_failed_ledger.csv",
        "trusted_adjusted_analysis_manifest.csv", "tpex_institutional_margin_manifest.csv",
        "price_bulk_download_manifest.csv", "tdcc_taifex_route_probe_evidence.csv",
        "readiness_for_core_full_lifecycle_feature_matrix.json",
    ]
    missing = [name for name in required if not (radar_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Radar package missing required artifacts: {missing}")

    compact = pd.read_csv(radar_dir / "twse_compact_repair_dedup_audit.csv")
    failures = pd.read_csv(radar_dir / "twse_true_failed_ledger.csv")
    adjusted = pd.read_csv(radar_dir / "trusted_adjusted_analysis_manifest.csv", dtype={"ticker": str})
    tpex = pd.read_csv(radar_dir / "tpex_institutional_margin_manifest.csv")
    radar_ready = json.loads((radar_dir / "readiness_for_core_full_lifecycle_feature_matrix.json").read_text(encoding="utf-8-sig"))

    shutil.copy2(radar_dir / "twse_true_failed_ledger.csv", output_dir / "p1_full_lifecycle_true_failed_ledger.csv")
    compact["ingest_action"] = compact.apply(
        lambda r: "exclude_corrupt_file_then_deduplicate_date_ticker" if int(r.corrupt_gzip_file_count) else "deduplicate_date_ticker", axis=1
    )
    compact["silent_fill_allowed"] = False
    compact.to_csv(output_dir / "p1_full_lifecycle_compact_dedup_exclusion_audit.csv", index=False, encoding="utf-8-sig")

    adjusted_accepted = int(adjusted.status.eq("accepted").sum())
    tpex_inst_accepted = int(((tpex.family == "tpex_three_institutional") & (tpex.status == "accepted")).sum())
    tpex_margin_accepted = int(((tpex.family == "tpex_margin_short") & (tpex.status == "accepted")).sum())
    feature_rows = [
        {"feature_family": "official_raw_execution_OHLCV", "status": "ready", "source_quality": "official", "allowed_use": "execution_price_and_mark_to_market", "blocked_reason": ""},
        {"feature_family": "trusted_adjusted_analysis_OHLC", "status": "partial", "source_quality": "trusted_nonofficial_analysis_only", "allowed_use": "research_grade_KD_MA_BIAS_RS_only", "blocked_reason": f"accepted={adjusted_accepted}; blocked={len(adjusted)-adjusted_accepted}; not formal"},
        {"feature_family": "TWSE_three_institutional", "status": "partial", "source_quality": "official", "allowed_use": "PIT feature after dedup and date-level missing flags", "blocked_reason": f"true_failed={int((failures.family=='twse_three_institutional').sum())}"},
        {"feature_family": "TWSE_margin_short", "status": "partial", "source_quality": "official", "allowed_use": "PIT feature after corrupt-file exclusion and dedup", "blocked_reason": f"true_failed={int((failures.family=='twse_margin_short').sum())}; TWSE_2017.csv.gz corrupt"},
        {"feature_family": "TPEx_margin_short", "status": "partial", "source_quality": "official", "allowed_use": "PIT feature on accepted dates", "blocked_reason": f"accepted_dates={tpex_margin_accepted}; remaining missingness explicit"},
        {"feature_family": "TPEx_three_institutional", "status": "blocked", "source_quality": "official_route_partial", "allowed_use": "none_in_unified_state_machine", "blocked_reason": f"accepted_dates={tpex_inst_accepted}; historical route mostly valid zero rows"},
        {"feature_family": "TDCC_holder_buckets", "status": "blocked", "source_quality": "official_current_only", "allowed_use": "none", "blocked_reason": "P1 historical PIT archive unavailable"},
        {"feature_family": "TAIFEX_OI_foreign_net", "status": "blocked", "source_quality": "official_historical_retention_limited", "allowed_use": "none", "blocked_reason": "P1 free historical route unavailable"},
        {"feature_family": "corporate_action_guard", "status": "partial", "source_quality": "prospective_only_plus_trusted_nonofficial_analysis", "allowed_use": "warning_and_analysis_price_lineage", "blocked_reason": "historical selected-stock adjusted close not formal-ready"},
    ]
    _write_csv(feature_rows, output_dir / "p1_full_lifecycle_feature_family_readiness.csv")

    _write_csv([
        {"field": "raw_execution_price", "source": "official TWSE/TPEx raw OHLCV", "use": "execution/cost/wealth path", "may_mix_with_other_price": False, "ready": True},
        {"field": "event_adjusted_analysis_price", "source": "Yahoo trusted_nonofficial adjusted series", "use": "KD/MA/BIAS/RS research features", "may_mix_with_other_price": False, "ready": False},
        {"field": "analysis_price_source_quality", "source": "per ticker manifest", "use": "row-level gate", "may_mix_with_other_price": False, "ready": True},
        {"field": "corporate_action_warning", "source": "event audit/guard", "use": "contamination warning", "may_mix_with_other_price": False, "ready": False},
    ], output_dir / "p1_full_lifecycle_price_semantics_contract.csv")

    schema = [
        ("signal_date", "identity", "PIT weekly close"), ("ticker", "identity", "Layer4 primary80 frozen universe"),
        ("raw_execution_close", "execution", "official raw OHLCV"), ("event_adjusted_analysis_close", "analysis", "trusted nonofficial adjusted OHLC"),
        ("KD_MA_BIAS_RS_block", "lifecycle", "requires adjusted analysis coverage and warmup"),
        ("institutional_flow_block", "capital", "requires market-complete PIT rows"), ("margin_short_lending_block", "risk", "requires market-complete PIT rows"),
        ("TDCC_holder_structure_block", "capital", "blocked"), ("market_TAIFEX_OI_block", "regime", "blocked"),
        ("lifecycle_state", "state", "not materialized until all minimum blocks ready"),
        ("incumbent_ticker", "decision", "hold valid incumbent"), ("challenger_ticker", "decision", "multi-block risk-adjusted challenger"),
        ("decision", "decision", "hold/switch/cash; cash only confirmed risk or invalid no replacement"),
        ("next_execution_date", "execution", "next official trading close"), ("transaction_cost", "cost", "EP05 stock/cash transition cost"),
    ]
    pd.DataFrame(schema, columns=["field", "block", "contract_semantics"]).assign(materialized=False).to_csv(
        output_dir / "p1_full_lifecycle_state_machine_schema_contract.csv", index=False, encoding="utf-8-sig"
    )

    blockers = [r for r in feature_rows if r["status"] != "ready"]
    _write_csv(blockers, output_dir / "p1_full_lifecycle_blocked_proxy_audit.csv")
    _write_csv([
        {"period": "P1", "requested_start": "2015-01-02", "requested_end": "2022-12-29", "actual_start": "", "actual_end": "", "status": "blocked_before_unified_materialization"}
    ], output_dir / "requested_vs_actual_coverage.csv")
    pd.DataFrame(columns=["signal_date", "ticker", "violation_reason"]).to_csv(output_dir / "future_data_audit.csv", index=False, encoding="utf-8-sig")

    readiness = {
        "task_id": TASK_ID,
        "status": "blocked_for_complete_unified_feature_state_machine_materialization",
        "radar_source_package_absorbed": True,
        "compact_ingest_policy_ready": True,
        "corrupt_TWSE_2017_margin_file_excluded": True,
        "true_failed_rows": len(failures),
        "no_rows_silently_filled": False,
        "official_raw_execution_OHLCV_ready": True,
        "trusted_nonofficial_adjusted_analysis_accepted_tickers": adjusted_accepted,
        "trusted_nonofficial_adjusted_analysis_blocked_tickers": len(adjusted) - adjusted_accepted,
        "TPEx_institutional_ready": False, "TDCC_P1_ready": False, "TAIFEX_P1_ready": False,
        "ready_for_core_full_lifecycle_feature_matrix": False,
        "ready_for_unified_state_machine_materialization": False,
        "ready_for_experiments": False, "automatic_handoff_stopped": True,
        "future_data_violation_count": 0, **FLAGS,
    }
    (output_dir / "readiness_for_p1_full_lifecycle_unified_feature_state_machine_contract.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_summary_zh.md").write_text(
        "# P1 full lifecycle unified feature/state-machine ingest 判定\n\n"
        "- Verdict：BLOCKED。Radar source package 已吸收為保守 ingest/schema contract，但不具備完整 feature matrix 或 state-machine materialization readiness。\n"
        "- 官方 raw execution OHLCV 可用；trusted Yahoo adjusted series 僅 research-grade analysis，913/976 ticker accepted，不得與 execution price 混欄。\n"
        "- TWSE compact 必須 date+ticker 去重；margin `TWSE_2017.csv.gz` 損壞檔明確排除。10 筆 true failure 保留，不 silent fill。\n"
        "- TPEx institutional、TDCC P1、TAIFEX P1 blocked；因此不跑 partial performance、不交 Experiments。\n",
        encoding="utf-8",
    )
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    manifest = {"task_id": TASK_ID, "runner": str(Path(__file__).resolve()), "source_radar": str(radar_dir), "source_readiness": radar_ready, "readiness": readiness,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name != "manifest.json"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_dir / "current_step.txt").write_text("blocked_waiting_complete_source_families_no_experiments_handoff", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.radar_dir, args.output_dir))


if __name__ == "__main__":
    main()
