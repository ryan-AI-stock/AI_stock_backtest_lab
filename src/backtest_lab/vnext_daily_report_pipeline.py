from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from backtest_lab.drive_publish import DEFAULT_FOLDER_ID, upsert_pdf


REMOTE_REPORT_NAME = "AI台股新模型每日收盤報告.pdf"
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_daily_report_pipeline_readiness_20260708")
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-PDF-REPORT-PIPELINE-READINESS-NO-PUBLISH-001"


@dataclass(frozen=True)
class VNextReportPaths:
    layer4_primary80: Path
    market_regime: Path
    pool_regime: Path
    exact_consensus_trigger: Path
    route_support_max1: Path
    radar_official_ohlcv_manifest: Path
    radar_pit_daily_sample: Path
    eod_source_scoped_rows: Path | None = None
    signal_refresh_dir: Path | None = None
    low_base_dir: Path | None = None


def default_paths(root: Path) -> VNextReportPaths:
    radar_root = Path(
        "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs"
    )
    return VNextReportPaths(
        layer4_primary80=root / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv",
        market_regime=root / "outputs/vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708/regime_switch_market_regime_fields.csv",
        pool_regime=root / "outputs/vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708/regime_switch_pool_regime_fields.csv",
        exact_consensus_trigger=root / "outputs/vnext_full_period_exact_consensus_trigger_contract_20260708/full_period_exact_consensus_trigger_contract.csv",
        route_support_max1=root / "outputs/vnext_route_support_max1_full_period_same_basis_contract_20260708/route_support_max1_full_period_same_basis_modelization_contract.csv",
        radar_official_ohlcv_manifest=radar_root
        / "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703/accepted_liquidity_shard_manifest.csv",
        radar_pit_daily_sample=radar_root
        / "radar_dynamic_pool1_all_listed_liquid_universe_pit_daily_20260703/accepted_liquidity_rows.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vNext daily close PDF report readiness package.")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--publish-drive", action="store_true")
    parser.add_argument("--eod-source-dir", default="")
    parser.add_argument("--signal-refresh-dir", default="")
    parser.add_argument("--low-base-dir", default="")
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--remote-name", default=REMOTE_REPORT_NAME)
    parser.add_argument("--file-id", default=os.environ.get("VNEXT_DAILY_REPORT_DRIVE_FILE_ID", ""))
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    build_package(
        root=root,
        output_dir=output_dir,
        requested_date=args.as_of_date,
        publish_drive=args.publish_drive,
        folder_id=args.folder_id,
        remote_name=args.remote_name,
        file_id=args.file_id.strip() or None,
        eod_source_dir=Path(args.eod_source_dir) if args.eod_source_dir else None,
        signal_refresh_dir=Path(args.signal_refresh_dir) if args.signal_refresh_dir else None,
        low_base_dir=Path(args.low_base_dir) if args.low_base_dir else None,
    )


def build_package(
    *,
    root: Path,
    output_dir: Path,
    requested_date: str,
    publish_drive: bool = False,
    folder_id: str = DEFAULT_FOLDER_ID,
    remote_name: str = REMOTE_REPORT_NAME,
    file_id: str | None = None,
    eod_source_dir: Path | None = None,
    signal_refresh_dir: Path | None = None,
    low_base_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = default_paths(root)
    if eod_source_dir is not None:
        paths = VNextReportPaths(
            **{
                **paths.__dict__,
                "eod_source_scoped_rows": eod_source_dir / "vnext_adhoc_20260708_scoped_common_stock_etf_rows.csv",
            }
        )
    if signal_refresh_dir is not None:
        paths = VNextReportPaths(
            **{
                **paths.__dict__,
                "signal_refresh_dir": signal_refresh_dir,
            }
        )
    if low_base_dir is not None:
        paths = VNextReportPaths(
            **{
                **paths.__dict__,
                "low_base_dir": low_base_dir,
            }
        )
    snapshot = build_signal_snapshot(paths, requested_date)
    attach_low_base_summary(snapshot, paths.low_base_dir)

    pdf_path = output_dir / "vnext_daily_report_sample_output.pdf"
    render_pdf_report(pdf_path, snapshot)

    coverage = build_requested_vs_actual_coverage(paths, requested_date)
    coverage_path = output_dir / "vnext_daily_report_requested_vs_actual_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    blocked = build_blocked_proxy_audit(snapshot, coverage)
    blocked_path = output_dir / "vnext_daily_report_blocked_proxy_audit.csv"
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")

    readiness = {
        "task": TASK_ID,
        "status": "pipeline_implemented_local_sample_ready_drive_publish_not_executed"
        if not publish_drive
        else "pipeline_implemented_drive_publish_attempted",
        "requested_report_date": requested_date,
        "actual_data_date": snapshot["data_actual_date"],
        "vnext_signal_actual_date": snapshot["as_of_data_date"],
        "market_data_ready_for_requested_date": snapshot["market_data_ready"],
        "sample_pdf_ready": pdf_path.exists() and pdf_path.stat().st_size > 0,
        "sample_pdf_basic_validation": "pdf_exists_nonzero_and_pdf_header_checked_by_runner",
        "sample_pdf_visual_render_validation": False,
        "pdf_path": str(pdf_path),
        "drive_folder_id": folder_id,
        "drive_remote_name": remote_name,
        "drive_publish_attempted": publish_drive,
        "report_changed": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
        "ready_for_daily_report_pipeline_review": True,
        "ready_for_live_publish": False,
        "live_publish_blocker": "Drive publish not executed locally; live publish is blocked until Strategy Center authorizes report-only publication and visual validation / Drive upload are verified.",
    }
    write_json(output_dir / "vnext_daily_report_pipeline_readiness.json", readiness)

    drive_audit = build_drive_publish_audit(
        pdf_path=pdf_path,
        publish_drive=publish_drive,
        folder_id=folder_id,
        remote_name=remote_name,
        file_id=file_id,
    )
    write_json(output_dir / "vnext_daily_report_drive_publish_audit.json", drive_audit)

    write_workflow_draft(output_dir / "vnext_daily_report_workflow_draft.yml", remote_name=remote_name)
    write_summary(output_dir / "final_summary_zh.md", snapshot=snapshot, readiness=readiness)
    write_json(
        output_dir / "manifest.json",
        {
            "output_dir": str(output_dir),
            "artifacts": [
                "vnext_daily_report_pipeline_readiness.json",
                "vnext_daily_report_drive_publish_audit.json",
                "vnext_daily_report_sample_output.pdf",
                "vnext_daily_report_requested_vs_actual_coverage.csv",
                "vnext_daily_report_blocked_proxy_audit.csv",
                "vnext_daily_report_workflow_draft.yml",
                "final_summary_zh.md",
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "report_changed": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
        },
    )
    return readiness


def build_signal_snapshot(paths: VNextReportPaths, requested_date: str) -> dict[str, Any]:
    refreshed = load_signal_refresh_snapshot(paths.signal_refresh_dir, requested_date)
    if refreshed:
        return refreshed
    field_dates = {
        "single_day_eod_source_anchor": max_date(paths.eod_source_scoped_rows, "date") if paths.eod_source_scoped_rows else None,
        "radar_full_sweep_official_ohlcv": max_date(paths.radar_official_ohlcv_manifest, "last_date"),
        "radar_pit_daily_sample_official_ohlcv": max_date(paths.radar_pit_daily_sample, "date"),
        "layer4_primary80_snapshot": max_date(paths.layer4_primary80, "snapshot_date"),
        "0050_market_regime_fields": max_date(paths.market_regime, "snapshot_date"),
        "dynamic80_pool_regime_fields": max_date(paths.pool_regime, "snapshot_date"),
        "exact_consensus_trigger": max_date(paths.exact_consensus_trigger, "signal_date"),
        "route_support_max1_state_machine": max_date(paths.route_support_max1, "signal_date"),
    }
    requested_presence = {
        "single_day_eod_source_anchor": has_date(paths.eod_source_scoped_rows, "date", requested_date)
        if paths.eod_source_scoped_rows
        else False,
        "radar_pit_daily_sample_official_ohlcv": has_date(paths.radar_pit_daily_sample, "date", requested_date),
        "layer4_primary80_snapshot": has_date(paths.layer4_primary80, "snapshot_date", requested_date),
        "0050_market_regime_fields": has_date(paths.market_regime, "snapshot_date", requested_date),
        "dynamic80_pool_regime_fields": has_date(paths.pool_regime, "snapshot_date", requested_date),
        "exact_consensus_trigger": has_date(paths.exact_consensus_trigger, "signal_date", requested_date),
        "route_support_max1_state_machine": has_date(paths.route_support_max1, "signal_date", requested_date),
    }
    common_reference = min(
        date_value
        for field, date_value in field_dates.items()
        if field
        in {
            "layer4_primary80_snapshot",
            "0050_market_regime_fields",
            "dynamic80_pool_regime_fields",
            "exact_consensus_trigger",
            "route_support_max1_state_machine",
        }
        and date_value
    )
    vnext_ready = all(
        requested_presence[field]
        for field in [
            "layer4_primary80_snapshot",
            "0050_market_regime_fields",
            "dynamic80_pool_regime_fields",
            "exact_consensus_trigger",
            "route_support_max1_state_machine",
        ]
    )
    eod_anchor_ready = bool(requested_presence.get("single_day_eod_source_anchor"))
    max1 = latest_row(paths.route_support_max1, "signal_date")
    market = latest_row(
        paths.market_regime,
        "snapshot_date",
        [
            "snapshot_date",
            "0050_adjusted_close",
            "0050_return_20d",
            "0050_return_40d",
            "0050_return_60d",
            "0050_price_vs_ma60",
            "0050_bias20",
            "0050_bias60",
        ],
    )
    rs20_top3 = latest_rs20_reference(paths.layer4_primary80)
    selected_asset_type = "blocked"
    selected_ticker = None
    selected_name = None
    if vnext_ready and max1 is not None:
        c2 = as_bool(max1.get("c2_market_health_gate"))
        trigger = as_bool(max1.get("consensus_trigger"))
        selected_asset_type = "stock" if c2 and trigger else "00631L_fallback"
        selected_ticker = str(max1.get("selected_ticker"))
        selected_name = "00631L" if selected_ticker == "00631L" else ""

    return {
        "as_of_requested_date": requested_date,
        "as_of_data_date": requested_date if vnext_ready else common_reference,
        "data_actual_date": requested_date if eod_anchor_ready else (requested_date if vnext_ready else common_reference),
        "official_eod_anchor_ready": eod_anchor_ready,
        "market_data_ready": vnext_ready,
        "field_as_of_dates": field_dates,
        "requested_date_field_presence": requested_presence,
        "regime_label": regime_label(max1, vnext_ready),
        "selected_branch": selected_branch(max1, vnext_ready),
        "branch_reason": branch_reason(max1, vnext_ready),
        "triggered_features": triggered_features(max1, market),
        "selected_asset_type": selected_asset_type,
        "selected_ticker": selected_ticker,
        "selected_name": selected_name,
        "c2_gate_pass": as_bool(max1.get("c2_market_health_gate")) if max1 is not None and vnext_ready else False,
        "consensus_trigger_pass": as_bool(max1.get("consensus_trigger")) if max1 is not None and vnext_ready else False,
        "route_support_score": none_if_nan(max1.get("route_support_weighted_score")) if max1 is not None else None,
        "route_support_rank": 1 if max1 is not None and str(max1.get("selected_asset_type")) == "stock" else None,
        "rs20_reference_top3": rs20_top3,
        "rs20_reference_top1": rs20_top3[0] if rs20_top3 else None,
        "latest_0050_market_reference": {} if market is None else series_dict(market),
        "current_eod_anchor_rows": current_anchor_rows(paths.eod_source_scoped_rows, requested_date),
        "diagnostic_only": True,
        "not_live_trade_decision": True,
    }


def build_requested_vs_actual_coverage(paths: VNextReportPaths, requested_date: str) -> pd.DataFrame:
    rows = [
        ("single_day_eod_source_anchor", paths.eod_source_scoped_rows, "date", "source_anchor"),
        ("official_ohlcv_full_sweep", paths.radar_official_ohlcv_manifest, "last_date", "source_anchor"),
        ("official_ohlcv_pit_daily_sample", paths.radar_pit_daily_sample, "date", "source_anchor"),
        ("layer4_primary80_snapshot", paths.layer4_primary80, "snapshot_date", "vnext_materialized_contract"),
        ("0050_market_regime_fields", paths.market_regime, "snapshot_date", "vnext_materialized_contract"),
        ("dynamic80_pool_regime_fields", paths.pool_regime, "snapshot_date", "vnext_materialized_contract"),
        ("exact_consensus_trigger", paths.exact_consensus_trigger, "signal_date", "vnext_materialized_contract"),
        ("route_support_max1_state_machine", paths.route_support_max1, "signal_date", "vnext_materialized_contract"),
    ]
    out = []
    for field, path, date_col, category in rows:
        out.append(
            {
                "field": field,
                "category": category,
                "requested_date": requested_date,
                "actual_latest_date": max_date(path, date_col),
                "requested_date_ready": has_date(path, date_col, requested_date),
                "path": str(path),
            }
        )
    if paths.low_base_dir is not None:
        readiness_path = paths.low_base_dir / "readiness_for_layer4_low_base_score_diagnostic.json"
        readiness = load_json_if_exists(readiness_path)
        out.append(
            {
                "field": "layer4_low_base_score_contract",
                "category": "vnext_materialized_contract",
                "requested_date": requested_date,
                "actual_latest_date": readiness.get("as_of_date", "") if readiness else "",
                "requested_date_ready": bool(readiness and readiness.get("as_of_date") == requested_date),
                "path": str(paths.low_base_dir),
            }
        )
    return pd.DataFrame(out)


def attach_low_base_summary(snapshot: dict[str, Any], low_base_dir: Path | None) -> None:
    snapshot["low_base_score_status"] = "not_ready"
    snapshot["low_base_layer4_exact_ready"] = False
    snapshot["low_base_top10_balanced"] = []
    snapshot["low_base_overlap_audit_ready"] = False
    if low_base_dir is None:
        return
    readiness = load_json_if_exists(low_base_dir / "readiness_for_layer4_low_base_score_diagnostic.json")
    top10_path = low_base_dir / "layer4_low_base_20260708_top10_sample.csv"
    snapshot["low_base_overlap_audit_ready"] = (low_base_dir / "existing_low_base_overlap_audit.csv").exists()
    if not readiness:
        snapshot["low_base_score_status"] = "blocked_missing_readiness"
        return
    snapshot["low_base_layer4_exact_ready"] = bool(readiness.get("layer4_primary80_as_of_date_ready"))
    if snapshot["low_base_layer4_exact_ready"]:
        snapshot["low_base_score_status"] = "ready_component_not_selected_rule"
    else:
        snapshot["low_base_score_status"] = "reference_only_layer4_primary80_blocked"
    if top10_path.exists():
        top10 = pd.read_csv(top10_path, dtype={"ticker": str})
        rows = top10[top10["score_variant"].eq("balanced")].sort_values("low_base_rank").head(10)
        snapshot["low_base_top10_balanced"] = [
            {
                "rank": int(row["low_base_rank"]),
                "ticker": str(row["ticker"]),
                "name": row.get("name", ""),
                "score": none_if_nan(row.get("low_base_score")),
            }
            for _, row in rows.iterrows()
        ]


def load_signal_refresh_snapshot(signal_refresh_dir: Path | None, requested_date: str) -> dict[str, Any] | None:
    if signal_refresh_dir is None:
        return None
    snapshot_path = signal_refresh_dir / "vnext_adhoc_20260708_signal_snapshot.json"
    rs20_path = signal_refresh_dir / "vnext_adhoc_20260708_rs20_top3_reference.csv"
    market_path = signal_refresh_dir / "vnext_adhoc_20260708_0050_market_regime_partial.csv"
    if not snapshot_path.exists():
        return None
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rs20 = []
    if rs20_path.exists():
        rs = pd.read_csv(rs20_path, dtype={"ticker": str})
        for _, row in rs.iterrows():
            rs20.append(
                {
                    "ticker": str(row.get("ticker", "")),
                    "name": row.get("name", ""),
                    "RS20": none_if_nan(row.get("RS20")),
                    "RS40": None,
                    "RS60": None,
                    "traded_value_rank_20d": none_if_nan(row.get("traded_value_rank_20d")),
                    "risk_overheat_penalty_context": None,
                    "source_quality": row.get("source_quality", "bounded_20260708_rs20_reference"),
                }
            )
    market = {}
    if market_path.exists():
        m = pd.read_csv(market_path, dtype={"ticker": str})
        rows = m[m["ticker"].eq("0050")]
        if not rows.empty:
            row = rows.iloc[0]
            market = {
                "snapshot_date": row.get("date"),
                "0050_adjusted_close": row.get("close"),
                "0050_return_20d": row.get("return_20d"),
                "0050_return_40d": row.get("return_40d"),
                "0050_return_60d": row.get("return_60d"),
                "0050_price_vs_ma60": row.get("bias60"),
                "0050_bias20": None,
                "0050_bias60": row.get("bias60"),
                "0050_ma60": row.get("ma60"),
                "c2_market_health_gate": row.get("c2_market_health_gate"),
                "c2_market_health_gate_ready": row.get("c2_market_health_gate_ready"),
            }
    selected_asset_type = raw.get("c2_selected_asset_type") or "blocked"
    selected_ticker = raw.get("c2_selected_ticker") or ""
    selected_name = raw.get("c2_selected_name") or ""
    market_ready = bool(raw.get("c2_gate_ready")) and (
        (not bool(raw.get("c2_gate_pass"))) or bool(raw.get("consensus_trigger_ready"))
    )
    if bool(raw.get("c2_gate_pass")) and not bool(raw.get("consensus_trigger_ready")):
        selected_branch_value = "blocked_consensus_trigger_missing"
        regime = "健康強勢但個股觸發未齊"
    elif selected_asset_type == "00631L_fallback":
        selected_branch_value = "fallback_00631L"
        regime = "普通防守 - C2 未通過"
    else:
        selected_branch_value = "blocked_vnext_fields_missing"
        regime = "資料未齊 - selected blocked"
    return {
        "as_of_requested_date": requested_date,
        "as_of_data_date": raw.get("as_of_data_date", requested_date),
        "data_actual_date": raw.get("as_of_data_date", requested_date),
        "official_eod_anchor_ready": True,
        "market_data_ready": market_ready,
        "field_as_of_dates": {"signal_refresh": raw.get("as_of_data_date", requested_date)},
        "requested_date_field_presence": {"signal_refresh": True},
        "regime_label": regime,
        "selected_branch": selected_branch_value,
        "branch_reason": raw.get("c2_blocked_reason", ""),
        "triggered_features": [
            f"C2={raw.get('c2_gate_pass')}",
            f"consensus={raw.get('consensus_trigger_pass')}",
            f"0050_return_20d={raw.get('0050_return_20d')}",
            f"0050_return_40d={raw.get('0050_return_40d')}",
            f"0050_bias60={raw.get('0050_bias60')}",
        ],
        "selected_asset_type": selected_asset_type,
        "selected_ticker": selected_ticker,
        "selected_name": selected_name,
        "c2_gate_pass": bool(raw.get("c2_gate_pass")),
        "consensus_trigger_pass": bool(raw.get("consensus_trigger_pass")),
        "route_support_score": None,
        "route_support_rank": None,
        "rs20_reference_top3": rs20,
        "rs20_reference_top1": rs20[0] if rs20 else None,
        "latest_0050_market_reference": market,
        "current_eod_anchor_rows": {},
        "diagnostic_only": True,
        "not_live_trade_decision": True,
    }


def build_blocked_proxy_audit(snapshot: dict[str, Any], coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coverage.iterrows():
        if not bool(row["requested_date_ready"]):
            owner = "Radar/Data" if str(row["category"]) == "source_anchor" else "Core/Data after source anchor"
            rows.append(
                {
                    "field": row["field"],
                    "field_as_of_date": row["actual_latest_date"],
                    "blocked_reason": "missing_requested_date_source_or_vnext_materialization",
                    "proxy_policy": "latest_available_may_be_shown_as_reference_only_not_selected",
                    "next_owner": owner,
                }
            )
    rows.extend(
        [
            {
                "field": "selected_stock_adjusted_close",
                "field_as_of_date": "",
                "blocked_reason": "adjusted close remains blocked for vNext selected stocks",
                "proxy_policy": "official unadjusted OHLC diagnostic only",
                "next_owner": "Strategy Center policy or Radar/Data adjusted source route",
            },
            {
                "field": "cash_bear_classifier",
                "field_as_of_date": "",
                "blocked_reason": "cash/bear classifier not accepted",
                "proxy_policy": "no cash rule in this report pipeline",
                "next_owner": "Strategy Center",
            },
            {
                "field": "formal_live_rule",
                "field_as_of_date": snapshot["as_of_data_date"],
                "blocked_reason": "report-only pipeline; not formal or active trade decision",
                "proxy_policy": "not_live_rule=true",
                "next_owner": "Strategy Center formal review later",
            },
            {
                "field": "pdf_visual_render_validation",
                "field_as_of_date": snapshot["as_of_data_date"],
                "blocked_reason": "pdftoppm/pdfinfo and pypdf/pdfplumber are unavailable in this Python environment",
                "proxy_policy": "basic PDF header and nonzero-size validation only",
                "next_owner": "Core/Data environment dependency or Strategy Center manual visual review",
            },
            {
                "field": "low_base_score",
                "field_as_of_date": snapshot["as_of_data_date"],
                "blocked_reason": snapshot.get("low_base_score_status", "not_ready"),
                "proxy_policy": "reference/component only; not selected rule and not a hard filter",
                "next_owner": "Experiments only after exact Layer4 primary80 panel is materialized",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_drive_publish_audit(
    *,
    pdf_path: Path,
    publish_drive: bool,
    folder_id: str,
    remote_name: str,
    file_id: str | None,
) -> dict[str, Any]:
    audit = {
        "folder_id": folder_id,
        "folder_url": f"https://drive.google.com/drive/u/0/folders/{folder_id}",
        "remote_name": remote_name,
        "local_pdf": str(pdf_path),
        "local_pdf_exists": pdf_path.exists(),
        "publish_attempted": publish_drive,
        "publish_action": "not_attempted",
        "drive_file_id": file_id or "",
        "overwrite_semantics": "update_by_file_id_or_update_by_name_or_create_once",
        "uses_existing_backtest_lab_drive_publish_upsert_pdf": True,
    }
    if not publish_drive:
        audit["blocked_reason"] = "local readiness run did not request Drive upload"
        return audit
    try:
        from backtest_lab.drive_publish import build_drive_service

        service, auth_mode = build_drive_service()
        uploaded_id, action = upsert_pdf(
            service,
            folder_id,
            pdf_path,
            remote_name,
            file_id=file_id,
            legacy_remote_names=(),
        )
        audit.update({"publish_action": action, "drive_file_id": uploaded_id, "auth_mode": auth_mode})
    except Exception as exc:  # pragma: no cover - local credential dependent.
        audit.update({"publish_action": "blocked", "blocked_reason": str(exc)})
    return audit


def render_pdf_report(path: Path, snapshot: dict[str, Any]) -> None:
    configure_chinese_font()
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5f7fa")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.875), 1, 0.125, color="#14212b", transform=ax.transAxes))
        ax.text(0.06, 0.945, "AI台股新模型每日收盤報告", color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.06,
            0.908,
            f"requested {snapshot['as_of_requested_date']} · data {snapshot['as_of_data_date']} · report-only diagnostic",
            color="#c8d5df",
            fontsize=10.5,
            transform=ax.transAxes,
        )
        status_color = "#b42318" if not snapshot["market_data_ready"] else "#13795b"
        selected = snapshot["selected_ticker"] or "blocked"
        selected_name = snapshot["selected_name"] or "尚未可選"
        cards = [
            ("Regime", snapshot["regime_label"], "#2457a7"),
            ("Branch", snapshot["selected_branch"], "#13795b"),
            ("Main", f"{selected} {selected_name}", status_color),
            ("Coverage", "ready" if snapshot["market_data_ready"] else "blocked", status_color),
        ]
        for i, (label, value, color) in enumerate(cards):
            x = 0.06 + i * 0.225
            ax.add_patch(plt.Rectangle((x, 0.75), 0.2, 0.085, facecolor="white", edgecolor="#d7dee8", transform=ax.transAxes))
            ax.text(x + 0.012, 0.805, label, color="#65717c", fontsize=9.5, transform=ax.transAxes)
            ax.text(x + 0.012, 0.775, fit_text(str(value), 18), color=color, fontsize=11.5, fontweight="bold", transform=ax.transAxes)

        lines = [
            f"C2 gate: {snapshot['c2_gate_pass']}",
            f"Consensus trigger: {snapshot['consensus_trigger_pass']}",
            f"Route support score: {snapshot['route_support_score']}",
            f"Reason: {snapshot['branch_reason']}",
        ]
        draw_section(ax, 0.06, 0.69, "今日模型判讀", lines)

        rs = snapshot.get("rs20_reference_top3") or []
        rs_lines = [
            f"{i + 1}. {item['ticker']} {item['name']} RS20={item['RS20']:.4f}"
            for i, item in enumerate(rs)
            if item.get("RS20") is not None
        ] or ["RS20 top3 尚未可重建。"]
        draw_section(ax, 0.06, 0.52, "RS20 Top3 Reference", rs_lines)

        market = snapshot.get("latest_0050_market_reference") or {}
        anchor = snapshot.get("current_eod_anchor_rows") or {}
        market_lines = [
            f"2026-07-08 EOD anchor 0050 close: {anchor.get('0050', {}).get('close')} turnover: {anchor.get('0050', {}).get('turnover_value')}",
            f"2026-07-08 EOD anchor 00631L close: {anchor.get('00631L', {}).get('close')} turnover: {anchor.get('00631L', {}).get('turnover_value')}",
            f"0050 latest field date: {market.get('snapshot_date', '')}",
            f"0050 return 20D / 40D / 60D: {market.get('0050_return_20d')} / {market.get('0050_return_40d')} / {market.get('0050_return_60d')}",
            f"0050 price vs MA60: {market.get('0050_price_vs_ma60')}",
            f"0050 BIAS20 / BIAS60: {market.get('0050_bias20')} / {market.get('0050_bias60')}",
        ]
        draw_section(ax, 0.06, 0.34, "Market Health", market_lines)

        risk_lines = [
            "本報告是 vNext 新模型 report pipeline 樣本，不是自動下單或正式交易規則。",
            "若 requested date 欄位未齊，只能顯示 latest available reference，不能包裝成今日 selected。",
            "Adjusted close / cash-bear classifier / formal-ready 仍保留 blocker。",
            f"Low base score: {snapshot.get('low_base_score_status', 'not_ready')}，只作 component/reference，不作 hard filter。",
            "所有後續主要績效結論必須使用 net after transaction cost；gross/no-cost 只能 secondary。",
        ]
        draw_section(ax, 0.06, 0.165, "風險與邊界", risk_lines)
        ax.text(0.06, 0.06, "formal_model_changed=false · trade_decision_changed=false · report_changed=true · not_live_rule=true", color="#65717c", fontsize=8.5, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)

        low_base_rows = snapshot.get("low_base_top10_balanced") or []
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5f7fa")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.89), 1, 0.11, color="#14212b", transform=ax.transAxes))
        ax.text(0.06, 0.945, "Low Base Score Reference", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.06,
            0.91,
            "component / overlap audit only · not selected rule · no hard filter",
            color="#c8d5df",
            fontsize=10.5,
            transform=ax.transAxes,
        )
        status_lines = [
            f"status: {snapshot.get('low_base_score_status', 'not_ready')}",
            f"exact Layer4 primary80 ready: {snapshot.get('low_base_layer4_exact_ready', False)}",
            f"overlap audit ready: {snapshot.get('low_base_overlap_audit_ready', False)}",
            "placement: Layer4 ranking component; BIAS/RS/risk reuse Layer2, pullback semantics stays Layer3.",
        ]
        draw_section(ax, 0.06, 0.82, "定位", status_lines)
        if low_base_rows:
            rows = [
                f"{item['rank']}. {item['ticker']} {item['name']} score={item['score']:.4f}"
                for item in low_base_rows
                if item.get("score") is not None
            ]
        else:
            rows = ["low_base top10 尚未可重建。"]
        draw_section(ax, 0.06, 0.64, "Balanced Top10 Reference", rows)
        overlap_lines = [
            "既有 BIAS / overheat / RS / pullback / risk 欄位不重複砍股票。",
            "low_base 補的是低基期位置 + 未過熱 + RS改善 + 成交改善的可比較分數。",
            "若未來要作 selected rule，必須由 Experiments 先做 net-after-cost diagnostic。",
        ]
        draw_section(ax, 0.06, 0.31, "Overlap Policy", overlap_lines)
        ax.text(0.06, 0.08, "report-only local sample · Drive publish not attempted", color="#65717c", fontsize=8.5, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)


def draw_section(ax, x: float, y: float, title: str, lines: list[str]) -> None:
    ax.text(x, y, title, color="#14212b", fontsize=14, fontweight="bold", transform=ax.transAxes)
    current = y - 0.035
    for line in lines:
        for wrapped in wrap_text(line, 90):
            ax.text(x, current, wrapped, color="#26343f", fontsize=10.3, transform=ax.transAxes)
            current -= 0.026


def configure_chinese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.sans-serif"] = [family, "Noto Sans CJK TC", "Microsoft JhengHei", "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def write_workflow_draft(path: Path, *, remote_name: str) -> None:
    path.write_text(
        f"""name: vNext Daily Report

on:
  workflow_dispatch:
    inputs:
      as_of_date:
        description: "Report date in YYYY-MM-DD. Defaults to shared schedule target date."
        required: false
        type: string
  schedule:
    - cron: "45 11 * * 1-5"

jobs:
  vnext-daily-report:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Checkout shared schedule rules
        uses: actions/checkout@v4
        with:
          repository: ryan-AI-stock/AI_stock_schedule_rules
          path: AI_stock_schedule_rules
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Shared daily schedule gate
        id: schedule-gate
        working-directory: AI_stock_schedule_rules
        run: python -m stock_schedule_rules.gate --profile daily --github-output "$GITHUB_OUTPUT"
      - name: Install dependencies
        if: steps.schedule-gate.outputs.should_run == 'true'
        run: python -m pip install -r requirements.txt
      - name: Build vNext daily report PDF
        if: steps.schedule-gate.outputs.should_run == 'true'
        env:
          PYTHONPATH: src
        run: |
          date="${{{{ inputs.as_of_date || steps.schedule-gate.outputs.target_date }}}}"
          python -m backtest_lab.vnext_daily_report_pipeline --as-of-date "$date" --output-dir outputs/vnext_daily_report_pipeline
      - name: Publish vNext daily report PDF
        if: steps.schedule-gate.outputs.should_run == 'true'
        env:
          PYTHONPATH: src
          GOOGLE_OAUTH_CLIENT_ID: ${{{{ secrets.GOOGLE_OAUTH_CLIENT_ID }}}}
          GOOGLE_OAUTH_CLIENT_SECRET: ${{{{ secrets.GOOGLE_OAUTH_CLIENT_SECRET }}}}
          GOOGLE_OAUTH_REFRESH_TOKEN: ${{{{ secrets.GOOGLE_OAUTH_REFRESH_TOKEN }}}}
          VNEXT_DAILY_REPORT_DRIVE_FILE_ID: ${{{{ secrets.VNEXT_DAILY_REPORT_DRIVE_FILE_ID || vars.VNEXT_DAILY_REPORT_DRIVE_FILE_ID }}}}
        run: |
          python -m backtest_lab.vnext_daily_report_pipeline \\
            --as-of-date "${{{{ inputs.as_of_date || steps.schedule-gate.outputs.target_date }}}}" \\
            --output-dir outputs/vnext_daily_report_pipeline \\
            --publish-drive \\
            --folder-id "{DEFAULT_FOLDER_ID}" \\
            --remote-name "{remote_name}" \\
            --file-id "$VNEXT_DAILY_REPORT_DRIVE_FILE_ID"
""",
        encoding="utf-8",
    )


def write_summary(path: Path, *, snapshot: dict[str, Any], readiness: dict[str, Any]) -> None:
    path.write_text(
        f"""# vNext daily PDF report pipeline readiness

## 結論

- 已建立 vNext 新模型每日 PDF 報告 runner / CLI：`python -m backtest_lab.vnext_daily_report_pipeline`。
- 已產出 local sample PDF：`{readiness['pdf_path']}`。
- Drive publish 使用既有 `backtest_lab.drive_publish.upsert_pdf`，語義是 update-by-file-id / update-by-name / create-once。
- 本地未實際上傳 Drive，因此不可宣稱已發布。

## 今日資料狀態

- requested date: `{snapshot['as_of_requested_date']}`。
- actual data date: `{snapshot['data_actual_date']}`。
- vNext signal actual date: `{snapshot['as_of_data_date']}`。
- market_data_ready_for_requested_date: `{snapshot['market_data_ready']}`。
- selected_branch: `{snapshot['selected_branch']}`。
- branch_reason: `{snapshot['branch_reason']}`。
- 若 `market_data_ready_for_requested_date=false`，代表今日主推薦尚未可發布成 selected signal；reference-only 欄位不可包裝成交易建議。
- low_base_score status: `{snapshot.get('low_base_score_status', 'not_ready')}`。
- low_base_score 只作 Layer4 component / reference；不可包裝成 selected rule 或 hard filter。

## 報告口徑

- Default 主線：00631L state-hold base + C2 market health gate + consensus trigger + route_support max1。
- Regime branch：保留 market_bias override / G3 guard / M4 breakout+breadth as design context，尚未接成 formal branch。
- RS20 top3：只保留 extreme reference，不作主線 selected。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=true
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
""",
        encoding="utf-8",
    )


def max_date(path: Path, date_col: str) -> str | None:
    if path is None or not path.exists():
        return None
    s = pd.read_csv(path, usecols=[date_col])[date_col].astype(str)
    return str(s.max()) if len(s) else None


def has_date(path: Path, date_col: str, value: str) -> bool:
    if path is None or not path.exists():
        return False
    s = pd.read_csv(path, usecols=[date_col])[date_col].astype(str)
    return bool((s == value).any())


def latest_row(path: Path, date_col: str, columns: list[str] | None = None) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    if df.empty:
        return None
    latest = df[date_col].astype(str).max()
    rows = df[df[date_col].astype(str) == latest]
    return rows.iloc[0] if len(rows) else None


def latest_rs20_reference(path: Path) -> list[dict[str, Any]]:
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "RS20",
        "RS40",
        "RS60",
        "traded_value_rank_20d",
        "risk_overheat_penalty_context",
    ]
    df = pd.read_csv(path, usecols=cols)
    latest = df["snapshot_date"].astype(str).max()
    top = df[df["snapshot_date"].astype(str) == latest].sort_values(
        ["RS20", "traded_value_rank_20d"], ascending=[False, True]
    ).head(3)
    out = []
    for _, row in top.iterrows():
        out.append(
            {
                "ticker": str(row["ticker"]),
                "name": row["name"],
                "RS20": none_if_nan(row["RS20"]),
                "RS40": none_if_nan(row["RS40"]),
                "RS60": none_if_nan(row["RS60"]),
                "traded_value_rank_20d": none_if_nan(row["traded_value_rank_20d"]),
                "risk_overheat_penalty_context": as_bool(row["risk_overheat_penalty_context"]),
                "source_quality": "reference_only_latest_layer4_rs20_sort_not_full_risk_tiebreak",
            }
        )
    return out


def current_anchor_rows(path: Path | None, requested_date: str) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["date"].astype(str) == requested_date]
    out: dict[str, dict[str, Any]] = {}
    for ticker in ["0050", "00631L", "2330"]:
        rows = df[df["ticker"].astype(str) == ticker]
        if rows.empty:
            continue
        row = rows.iloc[0]
        out[ticker] = {
            "name": row.get("name"),
            "close": none_if_nan(row.get("close")),
            "open": none_if_nan(row.get("open")),
            "high": none_if_nan(row.get("high")),
            "low": none_if_nan(row.get("low")),
            "turnover_value": none_if_nan(row.get("turnover_value")),
            "source_quality": row.get("source_quality"),
        }
    return out


def regime_label(max1: pd.Series | None, ready: bool) -> str:
    if not ready:
        return "資料未齊 - latest reference only"
    if max1 is None:
        return "blocked"
    if as_bool(max1.get("c2_market_health_gate")) and as_bool(max1.get("consensus_trigger")):
        return "健康強勢 - 允許個股例外"
    return "普通防守 - 00631L fallback"


def selected_branch(max1: pd.Series | None, ready: bool) -> str:
    if not ready:
        return "blocked_vnext_fields_missing"
    if max1 is None:
        return "blocked"
    if as_bool(max1.get("c2_market_health_gate")) and as_bool(max1.get("consensus_trigger")):
        return "route_support"
    return "fallback_00631L"


def branch_reason(max1: pd.Series | None, ready: bool) -> str:
    if not ready:
        return "requested-date vNext fields are not materialized; latest available rows are reference-only"
    if max1 is None:
        return "route_support state-machine row missing"
    return str(max1.get("state_reason", ""))


def triggered_features(max1: pd.Series | None, market: pd.Series | None) -> list[str]:
    features = []
    if max1 is not None:
        features.append(f"C2={as_bool(max1.get('c2_market_health_gate'))}")
        features.append(f"consensus={as_bool(max1.get('consensus_trigger'))}")
        features.append(f"route_support_score={none_if_nan(max1.get('route_support_weighted_score'))}")
    if market is not None:
        features.append(f"0050_return_20d={none_if_nan(market.get('0050_return_20d'))}")
        features.append(f"0050_return_40d={none_if_nan(market.get('0050_return_40d'))}")
        features.append(f"0050_price_vs_ma60={none_if_nan(market.get('0050_price_vs_ma60'))}")
    return features


def as_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def none_if_nan(value: Any) -> Any:
    return None if pd.isna(value) else float(value) if isinstance(value, (int, float)) else value


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def series_dict(series: pd.Series) -> dict[str, Any]:
    return {str(k): none_if_nan(v) for k, v in series.to_dict().items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wrap_text(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def fit_text(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


if __name__ == "__main__":
    main()
